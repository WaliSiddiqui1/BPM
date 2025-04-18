from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import os
import numpy as np
import random

class HowTo100M_DataLoader(tf.keras.utils.Sequence):
    
    def __init__(
            self,
            csv,
            features_path,
            data_dict,
            tokenizer,
            min_time=10.0,
            feature_framerate=1.0,
            max_words=30,
            min_words=0,
            n_pair=-1,
            max_frames=100,
            with_long_context=True,
            use_mil=False,
            only_sim=False,
            sampled_use_mil=False,
            pretrain_enhance_vmodal=False,
            video_dim=1024,
            batch_size=32
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.features_path = features_path
        self.data_dict = data_dict
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.min_words = min_words
        self.tokenizer = tokenizer
        self.n_pair = n_pair
        self.with_long_context = with_long_context
        self.feature_size = video_dim
        self.batch_size = batch_size

        self.only_sim = only_sim
        self.pretrain_enhance_vmodal = pretrain_enhance_vmodal
        self.iter_num = len(self.csv)

        self.use_mil = use_mil
        self.sampled_use_mil = sampled_use_mil
        if self.sampled_use_mil:
            self.use_mil = True

        if self.use_mil:
            positive_n_pair = self.n_pair
            video_id_list = [itm for itm in self.csv['video_id'].values]
            self.video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
            self.iter2video_pairs_dict = {}
            self.iter2video_pairslist_dict = {}
            iter_idx_mil_ = 0
            for video_id in video_id_list:
                data_dict = self.data_dict[video_id]
                n_caption = len(data_dict['start'])

                sub_list = []
                if self.n_pair < 0 or self.n_pair == 1:
                    for sub_id in range(n_caption):
                        sub_list.append([sub_id])
                else:
                    sb_ls_ = list(range(n_caption))
                    if self.n_pair > n_caption:
                        sb_ls_ = sb_ls_ * (self.n_pair // n_caption + 1)
                        sb_ls_ = sb_ls_[:self.n_pair]
                        for sub_id in np.arange(0, len(sb_ls_), self.n_pair):
                            sub_list.append(sb_ls_[sub_id: sub_id + self.n_pair])
                    else:
                        sb_ls_ = sb_ls_ + sb_ls_[:(((n_caption+positive_n_pair-1)//positive_n_pair)*positive_n_pair-n_caption)]
                        for sub_id in np.arange(0, len(sb_ls_), positive_n_pair):
                            pos_ls = sb_ls_[sub_id: sub_id + positive_n_pair]
                            sub_list.append(pos_ls)

                for sub_e in sub_list:
                    self.iter2video_pairs_dict[iter_idx_mil_] = (video_id, sub_e)
                    iter_idx_mil_ += 1
                self.iter2video_pairslist_dict[video_id] = sub_list

        if self.use_mil and self.sampled_use_mil is False:
            self.iter_num = len(self.iter2video_pairs_dict)

    def __len__(self):
        """
        returns num of batches in dataset
        """
        return (self.iter_num + self.batch_size - 1) // self.batch_size

    def _mask_tokens(self, words):
        """
        applies random masking to tokens for masked language modeling.
        
        this implements BERT-style masking where:
        - 15% of tokens are candidates for masking
        - 80% of those are replaced with [MASK]
        - 10% are replaced with random tokens
        - 10% remain unchanged
        
        args:
            words: list of tokens to be masked
            
        returns:
            masked_tokens: list of tokens with masking applied
            token_labels: labels for masked token prediction (-1 for unmasked positions)
        """
        token_labels = []
        masked_tokens = words.copy()

        for token_id, token in enumerate(masked_tokens):
            if token_id == 0 or token_id == len(masked_tokens) - 1:
                token_labels.append(-1)
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    masked_tokens[token_id] = "[MASK]"
                elif prob < 0.9:
                    masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                try:
                    token_labels.append(self.tokenizer.vocab[token])
                except KeyError:
                    token_labels.append(self.tokenizer.vocab["[UNK]"])
            else:
                token_labels.append(-1)

        return masked_tokens, token_labels

    def _get_text(self, video_id, n_pair_max, sub_ids=None, only_sim=False, enhance_vmodel=False):
        """
        process and prepare text data for given video
        """
        data_dict = self.data_dict[video_id]

        if self.use_mil:
            k = len(sub_ids)
            r_ind = sub_ids
        else:
            n_caption = len(data_dict['start'])
            if n_pair_max == -1:
                k = n_caption
                r_ind = range(n_caption)
            else:
                k = n_pair_max
                if k <= n_caption:
                    r_ind = np.random.choice(range(n_caption), k, replace=False)
                else:
                    r_ind_must = np.array(range(n_caption))
                    r_ind_rand = np.random.choice(range(n_caption), k-n_caption, replace=True)
                    r_ind = np.concatenate((r_ind_must, r_ind_rand), axis=0)
                np.random.shuffle(r_ind)

        starts = np.zeros(k)
        ends = np.zeros(k)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int32)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int32)

        for i in range(k):
            ind = r_ind[i]
            words, start_, end_ = self._get_single_transcript(data_dict, ind, with_long_context=self.with_long_context)
            caption_words = words.copy()
            starts[i], ends[i] = start_, end_

            if enhance_vmodel:
                words = []

            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]
            
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

            if only_sim is False:
                if len(caption_words) > total_length_with_CLS:
                    caption_words = caption_words[:total_length_with_CLS]
                input_caption_words = ["[CLS]"] + caption_words
                output_caption_words = caption_words + ["[SEP]"]

                masked_tokens, token_labels = self._mask_tokens(words)
                masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
                masked_input_caption_words, input_token_labels = self._mask_tokens(input_caption_words)
                input_caption_words = masked_input_caption_words.copy()

                while len(masked_token_ids) < self.max_words:
                    masked_token_ids.append(0)
                    token_labels.append(-1)
                assert len(masked_token_ids) == self.max_words
                assert len(token_labels) == self.max_words

                input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
                output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
                decoder_mask = [1] * len(input_caption_ids)
                while len(input_caption_ids) < self.max_words:
                    input_caption_ids.append(0)
                    output_caption_ids.append(0)
                    decoder_mask.append(0)
                assert len(input_caption_ids) == self.max_words
                assert len(output_caption_ids) == self.max_words
                assert len(decoder_mask) == self.max_words

                pairs_masked_text[i] = np.array(masked_token_ids)
                pairs_token_labels[i] = np.array(token_labels)

                pairs_input_caption_ids[i] = np.array(input_caption_ids)
                pairs_output_caption_ids[i] = np.array(output_caption_ids)
                pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, starts, ends

    def _get_single_transcript(self, data_dict, ind, with_long_context=True):
        """
        extract and potentially extend transcript for a single video segment.
        
        if with_long_context is enabled, this function will expand the context
        by including adjacent segments until meeting minimum requirements
        for time duration and word count.
        
        args:
            data_dict: dict containing transcript data
            ind: index of the current segment
            with_long_context: whether to extend context
            
        returns:
            words: list of tokenized words
            start: start time of the segment (possibly extended)
            end: end time of the segment (possibly extended)
        """
        start, end = ind, ind
        words = self.tokenizer.tokenize(str(data_dict['text'][ind]))
        diff = data_dict['end'][end] - data_dict['start'][start]
        while with_long_context and (len(words) < self.min_words or diff < self.min_time):
            if start > 0 and end < len(data_dict['end']) - 1:
                next_words = self.tokenizer.tokenize(str(data_dict['text'][end + 1]))
                prev_words = self.tokenizer.tokenize(str(data_dict['text'][start - 1]))
                d1 = data_dict['end'][end + 1] - data_dict['start'][start]
                d2 = data_dict['end'][end] - data_dict['start'][start - 1]
                if (self.min_time > 0 and d2 <= d1) or \
                    (self.min_time == 0 and len(next_words) <= len(prev_words)):
                    start -= 1
                    words = prev_words + words
                else:
                    end += 1
                    words.extend(next_words)
            elif start > 0:
                words = self.tokenizer.tokenize(str(data_dict['text'][start - 1])) + words
                start -= 1
            elif end < len(data_dict['end']) - 1:
                words.extend(self.tokenizer.tokenize(str(data_dict['text'][end + 1])))
                end += 1
            else:
                break
            diff = data_dict['end'][end] - data_dict['start'][start]
        return words, data_dict['start'][start], data_dict['end'][end]

    def _expand_video_slice(self, s, e, si, ei, fps, video_features):
        """
        extract and potentially expand a video slice based on timestamps.
        
        this handles edge cases where a video slice might be empty by
        adaptively expanding to adjacent frames.
        
        args:
            s: array of start timestamps
            e: array of end timestamps
            si: index in s to use
            ei: index in e to use
            fps: frames per second
            video_features: array of video features
            
        returns:
            video_slice: extracted video feature slice
            start: actual start frame index used
            end: actual end frame index used
        """
        start = int(s[si] * fps)
        end = int(e[ei] * fps) + 1

        if start > end:
            start, end = end, start
        video_slice = video_features[start:end]

        expand_left = True
        while len(video_slice) < 1:
            if si==0 and ei==len(s)-1:
                break
            if expand_left:
                expand_left = False
                si = si-1 if si>0 else si
            else:
                expand_left = True
                ei = ei+1 if ei<len(e)-1 else ei
            start = int(s[si] * fps)
            end = int(e[ei] * fps) + 1
            if start > end:
                start, end = end, start
            video_slice = video_features[start:end]

        if self.max_frames < video_slice.shape[0]:
            video_slice = video_slice[:self.max_frames]

        return video_slice, start, end

    def _get_video(self, idx, s, e, only_sim=False):
        """
        load and process video features for given segments.
        
        handles feature extraction, masking, and padding, including
        the masking procedure for masked frame modeling.
        
        args:
            idx: index in the CSV file
            s: array of start timestamps
            e: array of end timestamps
            only_sim: skip masked modeling preparations
            
        returns:
            video: tensor of video features
            video_mask: mask indicating valid frames
            masked_video: video with random frames masked
            video_labels_index: indices of masked frames
        """
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.int32)

        max_video_length = [0] * len(s)

        video = np.zeros((len(s), self.max_frames, self.feature_size), dtype=np.float32)
        feature_file = os.path.join(self.features_path, self.csv["feature_file"].values[idx])
        try:
            video_features = np.load(feature_file)

            for i in range(len(s)):
                if len(video_features) < 1:
                    raise ValueError("{} is empty.".format(feature_file))
                video_slice, start, end = self._expand_video_slice(s, e, i, i, self.feature_framerate, video_features)
                slice_shape = video_slice.shape
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
                if len(video_slice) < 1:
                    pass
                else:
                    video[i][:slice_shape[0]] = video_slice
        except Exception as e:
            print("video_id: {} error.".format(feature_file))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(s))]
        masked_video = video.copy()
        if only_sim is False:
            for i, video_pair_ in enumerate(masked_video):
                for j, _ in enumerate(video_pair_):
                    if j < max_video_length[i]:
                        prob = random.random()
                        if prob < 0.15:
                            masked_video[i][j] = [0.] * video.shape[-1]
                            video_labels_index[i].append(j)
                        else:
                            video_labels_index[i].append(-1)
                    else:
                        video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.int32)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def second_to_stamp(self, in_seconds):
        """
        convert seconds to timestamp str format
        """
        m, s = divmod(in_seconds, 60)
        h, m2 = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m2, s)

    def __getitem__(self, batch_idx):
        """
        get a batch of data at the specified index.
        
        this is the main function called during training to retrieve batches.
        handles all the preprocessing steps for both text and video,
        applying different strategies based on configuration.
        
        args:
            batch_idx: index of the batch to retrieve
            
        returns:
            tuple of tensors containing processed text and video data:
            - text inputs (tokens, masks, segments)
            - video features and masks
            - masked text and video for masked modeling tasks
            - caption generation inputs and targets
        """
        batch_start = batch_idx * self.batch_size
        batch_end = min((batch_idx + 1) * self.batch_size, self.iter_num)
        batch_size = batch_end - batch_start
        
        batch_pairs_text = []
        batch_pairs_mask = []
        batch_pairs_segment = []
        batch_video = []
        batch_video_mask = []
        batch_pairs_masked_text = []
        batch_pairs_token_labels = []
        batch_masked_video = []
        batch_video_labels_index = []
        batch_pairs_input_caption_ids = []
        batch_pairs_decoder_mask = []
        batch_pairs_output_caption_ids = []
        
        for feature_idx in range(batch_start, batch_end):
            if self.sampled_use_mil:
                idx = feature_idx
                video_id = self.csv['video_id'].values[idx]
                sub_list = self.iter2video_pairslist_dict[video_id]
                ranint = np.random.randint(0, len(sub_list))
                sub_ids = sub_list[ranint]
            elif self.use_mil:
                video_id, sub_ids = self.iter2video_pairs_dict[feature_idx]
                idx = self.video_id2idx_dict[video_id]
            else:
                idx = feature_idx
                video_id = self.csv['video_id'].values[idx]
                sub_ids = None

            enhance_vmodel = False
            if self.only_sim is False and self.pretrain_enhance_vmodal:
                prob = random.random()
                if prob < 0.15:
                    enhance_vmodel = True

            pairs_text, pairs_mask, pairs_segment, \
            pairs_masked_text, pairs_token_labels, pairs_input_caption_ids, \
            pairs_decoder_mask, pairs_output_caption_ids, \
            starts, ends = self._get_text(video_id, self.n_pair, sub_ids, only_sim=self.only_sim, enhance_vmodel=enhance_vmodel)

            video, video_mask, masked_video, video_labels_index = self._get_video(idx, starts, ends, only_sim=self.only_sim)
            
            batch_pairs_text.append(pairs_text)
            batch_pairs_mask.append(pairs_mask)
            batch_pairs_segment.append(pairs_segment)
            batch_video.append(video)
            batch_video_mask.append(video_mask)
            batch_pairs_masked_text.append(pairs_masked_text)
            batch_pairs_token_labels.append(pairs_token_labels)
            batch_masked_video.append(masked_video)
            batch_video_labels_index.append(video_labels_index)
            batch_pairs_input_caption_ids.append(pairs_input_caption_ids)
            batch_pairs_decoder_mask.append(pairs_decoder_mask)
            batch_pairs_output_caption_ids.append(pairs_output_caption_ids)
        
        batch_pairs_text = tf.convert_to_tensor(np.concatenate(batch_pairs_text, axis=0), dtype=tf.int32)
        batch_pairs_mask = tf.convert_to_tensor(np.concatenate(batch_pairs_mask, axis=0), dtype=tf.int32)
        batch_pairs_segment = tf.convert_to_tensor(np.concatenate(batch_pairs_segment, axis=0), dtype=tf.int32)
        batch_video = tf.convert_to_tensor(np.concatenate(batch_video, axis=0), dtype=tf.float32)
        batch_video_mask = tf.convert_to_tensor(np.concatenate(batch_video_mask, axis=0), dtype=tf.int32)
        batch_pairs_masked_text = tf.convert_to_tensor(np.concatenate(batch_pairs_masked_text, axis=0), dtype=tf.int32)
        batch_pairs_token_labels = tf.convert_to_tensor(np.concatenate(batch_pairs_token_labels, axis=0), dtype=tf.int32)
        batch_masked_video = tf.convert_to_tensor(np.concatenate(batch_masked_video, axis=0), dtype=tf.float32)
        batch_video_labels_index = tf.convert_to_tensor(np.concatenate(batch_video_labels_index, axis=0), dtype=tf.int32)
        batch_pairs_input_caption_ids = tf.convert_to_tensor(np.concatenate(batch_pairs_input_caption_ids, axis=0), dtype=tf.int32)
        batch_pairs_decoder_mask = tf.convert_to_tensor(np.concatenate(batch_pairs_decoder_mask, axis=0), dtype=tf.int32)
        batch_pairs_output_caption_ids = tf.convert_to_tensor(np.concatenate(batch_pairs_output_caption_ids, axis=0), dtype=tf.int32)
        
        return (batch_pairs_text, batch_pairs_mask, batch_pairs_segment, batch_video, batch_video_mask,
                batch_pairs_masked_text, batch_pairs_token_labels, batch_masked_video, batch_video_labels_index,
                batch_pairs_input_caption_ids, batch_pairs_decoder_mask, batch_pairs_output_caption_ids)