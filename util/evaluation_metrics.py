import numpy as np


class EvaluationMetrics:

    @staticmethod
    def calculate_oscr(gt, scores, unk_label=-1):
        """
        Calculates the OSCR values, iterating over the score of the target class of every sample,
        produces a pair (ccr, fpr) for every score.

        Args:
            gt (np.array): Integer array of target class labels.
            scores (np.array): Float array of dim [N_samples, N_classes]
            unk_label (int): Label to calculate the fpr, either negatives or unknowns. Defaults to -1 (negatives)

        Returns:
            Two lists: first one for ccr, second for fpr.
        """
        # Change the unk_label to calculate for kn_unknown or unk_unknown
        gt = gt.astype(int)
        kn = gt != unk_label
        unk = gt == unk_label

        # Get total number of samples of each type
        total_kn = np.sum(kn)
        total_unk = np.sum(unk)

        ccr, fpr = [], []
        # get predicted class for known sample

        pred_class = np.argmax(scores, axis=1)[kn]
        correctly_predicted = pred_class == gt[kn]
        target_score = scores[kn][range(kn.sum()), gt[kn]]

        # get maximum scores for unknown samples
        max_score = np.max(scores, axis=1)[unk]

        # Any max score can be a threshold
        # thresholds = np.unique(max_score)


        thresholds = np.unique(np.max(scores, axis=1))

        for tau in thresholds:
            # compute CCR value
            val = (correctly_predicted & (target_score >= tau)).sum() / total_kn
            ccr.append(val)

            val = (max_score >= tau).sum() / total_unk
            fpr.append(val)

        ccr = np.array(ccr)
        fpr = np.array(fpr)
        return ccr, fpr

    @staticmethod
    def ccr_at_fpr(gt, scores, unk_label=-1, fpr_values=[0.001,0.005,0.01,0.05,0.1,0.5]):
        """
        Calculates the Correct Classification Rate (CCR) at specified False Positive Rates (FPR) values.

        Args:
            gt (array): Ground truth labels.
            scores (array): Computed scores for each sample.
            fpr_values (list): List of FPR values at which to calculate the CCR.
            unk_label (int): Label used for unknown classes, default is -1.

        Returns:
            list: CCR values corresponding to each FPR threshold, or None if no valid CCR is found.
        """

        # compute ccr and fpr values from scores

        ccr, fpr = EvaluationMetrics.calculate_oscr(gt, scores, unk_label)

        ccrs = []
        for t in fpr_values:
            # get the FPR value that is closest, but above the current threshold
            candidates = np.nonzero(np.maximum(t - fpr, 0))[0]
            if candidates.size > 0:
                ccrs.append(round(ccr[candidates[0]]*100,2))
            else:
                ccrs.append(None)

        return  ccr, fpr,ccrs
