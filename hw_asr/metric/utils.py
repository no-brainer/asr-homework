import editdistance
# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0 and len(predicted_text) == 0:
        return 0.
    elif len(target_text) == 0:
        return 1.

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0 and len(predicted_text) == 0:
        return 0.
    elif len(target_text) == 0:
        return 1.

    target_text_words = target_text.split()
    return editdistance.eval(target_text_words, predicted_text.split()) / len(target_text_words)
