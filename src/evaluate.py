import evaluate

# load HuggingFace-compatible evaluation metrics
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
cider = evaluate.load("cider")
spice = evaluate.load("spice")

# evaluate a list of predictions against reference captions
def evaluate_all_metrics(preds, refs):
    """
    Args:
        preds: list of strings — model-generated captions
        refs: list of strings — ground truth captions
    Returns:
        dict with metric scores
    """
    
    # BLEU and CIDEr expect list-of-list references
    refs_list = [[r] for r in refs]

    return {
        "BLEU": bleu.compute(predictions=preds, references=refs_list),
        "METEOR": meteor.compute(predictions=preds, references=refs),
        "ROUGE": rouge.compute(predictions=preds, references=refs),
        "CIDEr": cider.compute(predictions=preds, references=refs_list),
        "SPICE": spice.compute(predictions=preds, references=refs_list),
    }