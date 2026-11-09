from sklearn.svm import SVC


def make_svm(kernel: str, c: float) -> SVC:
    return SVC(C=c, kernel=kernel, gamma="scale", class_weight="balanced")
