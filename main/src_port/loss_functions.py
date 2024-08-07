import torch




# ---------------------------------------
# >> REPLUG


def compute_perplexity(output, labels, pad_token_id):
    logits = output["logits"]  # [bs, seq_len, vocab_size]

    # Ensure labels are within the valid range of indices
    assert torch.all(labels < logits.size(-1)), "Labels contain indices out of range for logits"

    #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [bs, seq_len, vocab_size]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)

    # Apply gather to get the log probabilities for the correct labels
    x = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))  # [bs, seq_len, 1]
    x = x.squeeze(-1)  # [bs, seq_len]

    # Create a mask for non-pad tokens
    mask = (labels != pad_token_id).float()  # [bs, seq_len]

    # Compute the mean log probability only for non-pad tokens
    masked_x = x * mask  # [bs, seq_len]
    mean_log_prob = masked_x.sum() / mask.sum()  # scalar

    return -mean_log_prob




# ----------------------------------------
# >> ORPO

# ORPO Authors: Jiwoo Hong, Noah Lee, and James Thorne
# Official code: https://github.com/xfactlab/orpo
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


def odds_ratio_loss(positive_retr_log_prob, 
                    positive_retr_log_prob, 
                    beta : float = 0.1):
    """
    Return the odd ration between input posiitve and negatative probabilities
    """
                
    # log ration reformulated as difference to enhance statbility
    log_odds = (positive_retr_log_prob - positive_retr_log_prob) - (
        torch.log1p(-torch.exp(positive_retr_log_prob)) - torch.log1p(-torch.exp(positive_retr_log_prob))
    )

    sig_ratio = F.sigmoid(log_odds)
    ratio = torch.log(sig_ratio)
    losses = beta * ratio

    positive_rewards = beta * positive_retr_log_prob.detach()
    negative_rewards = beta * positive_retr_log_prob.detach()

    return losses, positive_rewards, negative_rewards, torch.mean(ratio).item(), torch.mean(log_odds).item()