#!/usr/bin/env python3
# 41729eed-1c9d-11e8-9de3-00505601122b
# 4d4a7a09-1d33-11e8-9de3-00505601122b
# 80f6d138-1c94-11e8-9de3-00505601122b

# Transformer architektura s relative positional representation a dvěma enkodéry -- jeden zpracovává písmena ve slově, druhý slova ve větě


import math
import numpy as np
import torch
import torch.nn as nn
from morpho_dataset import MorphoDataset


def arrange_char_pos_embedding(len_k, len_q, max_len, embedding):
    k = torch.arange(len_k, device='cuda')
    q = torch.arange(len_q, device='cuda')
    indices = k.view(1, -1) - q.view(-1, 1)
    indices.clamp_(-max_len, max_len).add_(max_len)

    return embedding[indices, :]

def arrange_word_pos_embedding(len_k, lens_q, max_len, embedding):
    k = torch.arange(len_k, device='cuda')
    q = torch.repeat_interleave(lens_q - lens_q.cumsum(dim=0), lens_q, dim=0) + torch.arange(lens_q.sum(), device='cuda')
    indices = k.view(1, -1) - q.view(-1, 1)
    indices.clamp_(-max_len, max_len).add_(max_len)

    return embedding[indices, :]

class UniformAttentionSublayer(nn.Module):
    def __init__(self, dimension, heads, max_pos_len):
        super(UniformAttentionSublayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.max_pos_len = max_pos_len
        self.scale = math.sqrt(dimension / heads)

        self.pos_embedding = torch.randn(2*max_pos_len + 1, dimension // heads, device='cuda', requires_grad=True)
        self.input_transform = nn.Linear(dimension, 3*dimension)
        self.softmax = nn.Softmax(dim=3)
        self.output_transform = nn.Linear(dimension, dimension)

    def forward(self, input, mask):
        batch_size, seq_len, _ = input.size()

        QKV = self.input_transform(input)
        QKV = QKV.view(batch_size, -1, self.heads, 3*self.dimension // self.heads)
        QKV = QKV.permute(0, 2, 1, 3).contiguous()
        Q, K, V = QKV.chunk(3, dim=3)

        logits = torch.matmul(Q, K.transpose(2, 3))

        arranged_pos = arrange_char_pos_embedding(seq_len, seq_len, self.max_pos_len, self.pos_embedding)
        Q_t = Q.permute(2, 0, 1, 3).view(seq_len, batch_size*self.heads, -1)
        pos_logits = torch.matmul(Q_t, arranged_pos.transpose(1, 2))
        pos_logits = pos_logits.view(seq_len, batch_size, self.heads, -1).permute(1, 2, 0, 3)
        logits = (logits + pos_logits) / self.scale

        if mask is not None: logits.masked_fill_(mask, -np.inf)
        indices = self.softmax(logits)
        combined = torch.matmul(indices, V)

        heads_concat = combined.permute(0, 2, 1, 3).contiguous()
        heads_concat = heads_concat.view(heads_concat.size(0), -1, self.dimension)

        return self.output_transform(heads_concat)


class DividedAttentionSublayer(nn.Module):
    def __init__(self, dimension, heads, max_pos_len):
        super(DividedAttentionSublayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.max_pos_len = max_pos_len
        self.scale = math.sqrt(dimension / heads)

        self.pos_embedding = torch.randn(2*max_pos_len + 1, dimension // heads, device='cuda', requires_grad=True)
        self.input_transform_q = nn.Linear(dimension, dimension)
        self.input_transform_k = nn.Linear(dimension, dimension)
        self.input_transform_v = nn.Linear(dimension, dimension)
        self.softmax = nn.Softmax(dim=3)
        self.output_transform = nn.Linear(dimension, dimension)

    def _split_heads(self, x):
        x = x.view(x.size(0), -1, self.heads, self.dimension // self.heads)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x_q, x_k, x_v, mask, sentence_lengths=None):
        Q = self._split_heads(self.input_transform_q(x_q))
        K = self._split_heads(self.input_transform_k(x_k))
        V = self._split_heads(self.input_transform_v(x_v))

        batch_size, _, len_q, _ = Q.size()
        len_k = K.size(2)

        logits = torch.matmul(Q, K.transpose(2, 3))

        if sentence_lengths is None:
            arranged_pos = arrange_char_pos_embedding(len_k, len_q, self.max_pos_len, self.pos_embedding)
            Q_t = Q.permute(2, 0, 1, 3).view(len_q, batch_size * self.heads, -1)
            pos_logits = torch.matmul(Q_t, arranged_pos.transpose(1, 2))
            pos_logits = pos_logits.view(len_q, batch_size, self.heads, -1).permute(1, 2, 0, 3)
        else:
            arranged_pos = arrange_word_pos_embedding(len_k, sentence_lengths, self.max_pos_len, self.pos_embedding)
            pos_logits = torch.matmul(Q.transpose(0, 1), arranged_pos.transpose(1, 2))
            pos_logits = pos_logits.transpose(0, 1)

        logits = (logits + pos_logits) / self.scale
        if mask is not None: logits.masked_fill_(mask, -np.inf)
        indices = self.softmax(logits)
        combined = torch.matmul(indices, V)

        heads_concat = combined.permute(0, 2, 1, 3).contiguous()
        heads_concat = heads_concat.view(heads_concat.size(0), -1, self.dimension)

        return self.output_transform(heads_concat)


class EncoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout, max_pos_len):
        super(EncoderLayer, self).__init__()

        self.attention = UniformAttentionSublayer(dimension, heads, max_pos_len)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(dimension)

        self.nonlinear_sublayer = nn.Sequential(
            nn.Linear(dimension, 4*dimension),
            nn.ReLU(),
            nn.Linear(4*dimension, dimension),
            nn.Dropout(dropout)
        )
        self.layer_norm_2 = nn.LayerNorm(dimension)

    def forward(self, x, mask):
        attention = self.attention(x, mask)
        attention = self.dropout_1(attention)
        x = self.layer_norm_1(attention + x)

        nonlinear = self.nonlinear_sublayer(x)
        return self.layer_norm_2(nonlinear + x)


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads, dropout, max_pos_len):
        super(DecoderLayer, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.dropout = dropout

        self.attention_1 = UniformAttentionSublayer(dimension, heads, max_pos_len)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(dimension)

        self.attention_2 = DividedAttentionSublayer(dimension, heads, max_pos_len)
        self.dropout_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(dimension)

        self.attention_3 = DividedAttentionSublayer(dimension, heads, max_pos_len)
        self.dropout_3 = nn.Dropout(dropout)
        self.layer_norm_3 = nn.LayerNorm(dimension)

        self.nonlinear_sublayer = nn.Sequential(
            nn.Linear(dimension, 4*dimension),
            nn.ReLU(),
            nn.Linear(4*dimension, dimension),
            nn.Dropout(dropout)
        )
        self.layer_norm_4 = nn.LayerNorm(dimension)

    def forward(self, char_encoder_output, word_encoder_output, self_input, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths):
        attention = self.attention_1(self_input, look_ahead_mask)
        attention = self.dropout_1(attention)
        x = self.layer_norm_1(attention + self_input)

        attention = self.attention_2(x, char_encoder_output, char_encoder_output, char_padding_mask)
        attention = self.dropout_2(attention)
        x = self.layer_norm_2(attention + x)

        attention = self.attention_3(x, word_encoder_output, word_encoder_output, word_padding_mask, sentence_lengths)
        attention = self.dropout_3(attention)
        x = self.layer_norm_3(attention + x)

        nonlinear = self.nonlinear_sublayer(x)
        return self.layer_norm_3(nonlinear + x)


class Encoder(nn.Module):
    def __init__(self, dimension, heads, layers, dropout, max_pos_len):
        super(Encoder, self).__init__()

        self.scale = float(math.sqrt(dimension))
        self.dropout = nn.Dropout(dropout)

        self.encoding = nn.ModuleList([EncoderLayer(dimension, heads, dropout, max_pos_len) for _ in range(layers)])

    def forward(self, x, mask):
        x = x * self.scale
        x = self.dropout(x)

        for encoding in self.encoding:
            x = encoding(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_chars, dimension, heads, layers, dropout, max_pos_len):
        super(Decoder, self).__init__()

        self.scale = float(math.sqrt(dimension))

        self.embedding = nn.Embedding(num_chars, dimension, padding_idx=MorphoDataset.Factor.PAD)
        self.dropout = nn.Dropout(dropout)
        self.char_dropout = nn.Dropout2d(dropout)

        self.decoding = nn.ModuleList([DecoderLayer(dimension, heads, dropout, max_pos_len) for _ in range(layers)])

    def forward(self, char_encoder_output, word_encoder_output, x, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths):
        x = self.embedding(x) * self.scale
        x = self.dropout(x)

        char_encoder_output = self.char_dropout(char_encoder_output.unsqueeze(0)).squeeze_(0)

        for decoding in self.decoding:
            x = decoding(char_encoder_output, word_encoder_output, x, look_ahead_mask, char_padding_mask, word_padding_mask, sentence_lengths)
        return x


class WordEmbedding(nn.Module):
    def __init__(self, args, num_source_chars):
        super().__init__()

        self.dim = args.dim
        self.embedding = nn.Embedding(num_source_chars, args.dim, padding_idx=MorphoDataset.Factor.PAD)

        convolutions = []
        for width in range(3, args.cnn_max_width + 1, 2):
            conv = [
                nn.Conv1d(args.dim, args.cnn_filters, kernel_size=width, stride=1, padding=(width - 1) // 2),
                nn.ReLU()
            ]
            for _ in range(1, args.cle_layers):
                conv.append(nn.Conv1d(args.cnn_filters, args.cnn_filters, kernel_size=width, stride=1, padding=(width-1)//2))
                conv.append(nn.ReLU())
            convolutions.append(nn.Sequential(*conv))

        self.convolutions = nn.ModuleList(convolutions)
        self.conv_linear = nn.Linear((args.cnn_max_width - 1)//2*args.cnn_filters, args.dim)
        self.conv_relu = nn.ReLU()
        self.combined_linear = nn.Linear(args.dim + MorphoDataset.Dataset.EMBEDDING_SIZE, args.dim)
        self.combined_relu = nn.ReLU()

    def forward(self, chars, word_embeddings, char_mask):
        sentences, words, char_len = chars.size()
        char_embedding = self.embedding(chars)
        char_embedding_flat = char_embedding.view(-1, char_len, self.dim).transpose(-2, -1)

        to_concat = []
        for convolution in self.convolutions:
            convoluted = convolution(char_embedding_flat)
            to_concat.append(convoluted.max(dim=-1).values)
        concated = torch.cat(to_concat, dim=-1)
        convoluted_char_embedding = self.conv_relu(self.conv_linear(concated)).view(sentences, words, -1)

        full_embedding = self.combined_linear(torch.cat([word_embeddings, convoluted_char_embedding], dim=-1))
        full_embedding = self.combined_relu(full_embedding)

        return chars[char_mask,:], char_embedding[char_mask,:,:], full_embedding


class Model(nn.Module):
    def __init__(self, args, num_source_chars, num_target_chars):
        super().__init__()

        self._input_embedding = WordEmbedding(args, num_source_chars)
        self._encoder = Encoder(args.dim, args.heads, args.layers, args.dropout, args.max_pos_len)
        self._encoder_sentence = Encoder(args.dim, args.heads, args.layers, args.dropout, args.max_pos_len)
        self._decoder = Decoder(num_target_chars, args.dim, args.heads, args.layers, args.dropout, args.max_pos_len)
        self._classifier = nn.Linear(args.dim, num_target_chars)

    def _create_look_ahead_mask(self, target):
        look_ahead = torch.ones(target.size(1), target.size(1), device='cuda', dtype=torch.uint8).triu(1)
        padding = self._create_padding_mask(target)
        return torch.max(padding, look_ahead)

    def _create_padding_mask(self, seq):
        seq = seq == 0
        seq.unsqueeze_(1).unsqueeze_(1)
        return seq

    def _gather_batch(self, batch, dataset):
        source_charseq_ids = torch.LongTensor(batch[dataset.FORMS].charseq_ids)
        source_charseqs = torch.LongTensor(batch[dataset.FORMS].charseqs)
        source_word_ids = torch.LongTensor(batch[dataset.FORMS].word_ids)
        source_words = torch.FloatTensor(batch[dataset.FORMS].word_embeddings)
        target_charseq_ids = torch.LongTensor(batch[dataset.LEMMAS].charseq_ids)
        target_charseqs = torch.LongTensor(batch[dataset.LEMMAS].charseqs)
        source_mask = source_charseq_ids != 0

        sources = source_charseqs[source_charseq_ids, :]
        targets = target_charseqs[target_charseq_ids, :][target_charseq_ids != 0]
        encoder_mask = self._create_padding_mask(sources[source_mask])

        sentence_lenghts = (source_word_ids != 0).sum(dim=1)
        #sentences = torch.repeat_interleave(source_words, sentence_lenghts, dim=0)
        encoder_sentence_mask = self._create_padding_mask(source_word_ids)
        #encoder_sentence_mask = torch.repeat_interleave(encoder_sentence_mask, sentence_lenghts, dim=0)

        return sources.cuda(), source_mask.cuda(), targets.cuda(), source_words.cuda(), encoder_mask.cuda(), encoder_sentence_mask.cuda(), sentence_lenghts.cuda()

    def forward(self, batch, dataset):
        sources, source_mask, targets, sentences, char_encoder_mask, word_encoder_mask, sentence_lengths = self._gather_batch(batch, dataset)
        targets_in, targets_out = targets[:, :-1], targets[:, 1:]

        decoder_combined_mask = self._create_look_ahead_mask(targets_in)

        sources, embedded_chars, embedded_words = self._input_embedding(sources, sentences, source_mask)
        encoded_chars = self._encoder(embedded_chars, char_encoder_mask)

        encoded_words = self._encoder_sentence(embedded_words, word_encoder_mask)
        encoded_words = torch.repeat_interleave(encoded_words, sentence_lengths, dim=0)
        word_encoder_mask = torch.repeat_interleave(word_encoder_mask, sentence_lengths, dim=0)

        decoded_chars = self._decoder(encoded_chars, encoded_words, targets_in, decoder_combined_mask, char_encoder_mask, word_encoder_mask, sentence_lengths)
        prediction = self._classifier(decoded_chars)

        return prediction, sources, targets_out

    def predict(self, batch, dataset):
        sources, source_mask, targets, sentences, char_encoder_mask, word_encoder_mask, sentence_lengths = self._gather_batch(batch, dataset)
        maximum_iterations = sources.size(1) + 10

        sources, embedded_chars, embedded_words = self._input_embedding(sources, sentences, source_mask)
        encoded_chars = self._encoder(embedded_chars, char_encoder_mask)

        encoded_words = self._encoder_sentence(embedded_words, word_encoder_mask)
        encoded_words = torch.repeat_interleave(encoded_words, sentence_lengths, dim=0)
        word_encoder_mask = torch.repeat_interleave(word_encoder_mask, sentence_lengths, dim=0)

        output = torch.full((encoded_chars.size(0), 1), MorphoDataset.Factor.BOW, device='cuda', dtype=torch.long)
        finished = torch.full((encoded_chars.size(0),), False, device='cuda', dtype=torch.uint8)

        for _ in range(maximum_iterations):
            decoder_combined_mask = self._create_look_ahead_mask(output)
            decoded_chars = self._decoder(encoded_chars, encoded_words, output, decoder_combined_mask, char_encoder_mask, word_encoder_mask, sentence_lengths)
            predictions = self._classifier(decoded_chars)

            next_prediction = predictions[:, -1, :]
            next_char = next_prediction.argmax(dim=-1)

            output = torch.cat((output, next_char.unsqueeze(-1)), -1)
            finished |= next_char == MorphoDataset.Factor.EOW

            if finished.all(): break

        return output[:, 1:], sources, targets, sentence_lengths

    def predict_to_list(self, batch, dataset):
        sentences = []

        predictions, _, _, sentence_lengths = self.predict(batch, dataset)
        predictions = predictions.cpu()

        index = 0
        for len in sentence_lengths:
            sentence = []
            for w in range(len.item()):
                word = []
                for prediction in predictions[index, :].numpy():
                    if prediction == MorphoDataset.Factor.EOW: break
                    word.append(prediction)
                word.append(MorphoDataset.Factor.EOW)

                index += 1
                sentence.append(word)
            sentences.append(sentence)

        return sentences