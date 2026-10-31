"""Prompts shared by data prep, training and evaluation, so all three stay identical."""

SYSTEM = "You are an expert listener in a subjective audio quality test."

# GRPO: reasoning in <think>, score in <answer> (VQAThinker format)
GRPO_PROMPT = (
    "Listen to the audio clip and judge its overall perceived quality on the ACR scale: "
    "1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent. "
    "First think about the quality problems you hear, if any (for example noise, distortion, "
    "coloration, band-limitation, dropouts, unnatural voice), inside <think> </think>. "
    "Then give the score as a number from 1.00 to 5.00 with two decimals inside "
    "<answer> </answer>."
)

# SFT baseline: score only, since we have no reasoning text to supervise
SFT_PROMPT = (
    "Listen to the audio clip and judge its overall perceived quality on the ACR scale: "
    "1 = bad, 2 = poor, 3 = fair, 4 = good, 5 = excellent. "
    "Reply with only the score as a number from 1.00 to 5.00 with two decimals inside "
    "<answer> </answer>."
)
