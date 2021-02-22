import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching

from paradigm_types import Form, Paradigm
from argparse import ArgumentParser
from typing import Dict, Generator, List, Set, Tuple


def iter_paradigms(fn: str) -> Generator[Paradigm, None, None]:
    """Read lines into Paradigms of Forms, yielding them as a Generator

    See: paradigm_types.py for reference"""
    with open(fn) as f:
        p = Paradigm()
        for line in f:
            # Line breaks indicate a new paradigm
            line = line.strip()
            if len(line) > 0:
                form = Form.from_line(line)
                # We do not want duplicate forms as
                # syncretic forms only need to be accounted for once.
                if form not in p.forms:
                    p.forms.append(form)
            else:
                yield p
                p = Paradigm()

        # Account for possible extra newline
        if len(p) > 0:
            yield p


class Evaluator:
    def __init__(self, ref_dict: Dict[str, Set[str]]):
        self.reference_dict = ref_dict

    def _get_metrics(self, true_pos: int, prd_size: int, ref_size: int) -> Tuple[float]:
        """Compute precision, recall, and F1"""
        precision = true_pos / prd_size if prd_size > 0 else 0
        recall = true_pos / ref_size if ref_size > 0 else 0
        f1 = 2 * (precision * recall/ (precision + recall)) if precision + recall > 0 else 0

        return precision, recall, f1

    def _prune_preds(self, pred_dict: Dict[str, Set[str]], ref: Set) -> Dict:
        """Remove predicted words that are not in the gold set.
        This means that extra words in the corpus that will not be evaluated on
        are not considered in evaluation."""
        pruned = {}
        for k, pred_words in pred_dict.items():
            words = set([w for w in pred_words if w in ref])
            if len(words) > 0:
                pruned[k] = words

        return pruned

    def score(self, pred_dict: Dict[str, Set[str]]) -> Dict:
        """Score the predicted paradigm clusters
        Return a dictionary whose keys are evaluation metrics from {precision, recall, f1}"""
        # First filter out pred words that are not in ref
        all_ref_set = set()
        all_ref_set.update(*self.reference_dict.values())
        # Remove the predicted bible words that are not in the gold set
        pred_dict = self._prune_preds(pred_dict, all_ref_set)

        graph = nx.Graph()
        pred_nodes = frozenset(pred_dict.keys())
        ref_nodes = frozenset(self.reference_dict.keys())
        graph.add_nodes_from(pred_nodes)
        graph.add_nodes_from(ref_nodes)

        # 2. get pairwise f1's
        ref_size = sum([len(r) for r in self.reference_dict.values()])
        for ref_k, ref_set in self.reference_dict.items():
            for pred_k, pred_set in pred_dict.items():
                # TP are the intersection of the predicted words, and gold words in a cluster
                true_pos = len(ref_set & pred_set)

                if true_pos > 0:
                    _, _, f1 = self._get_metrics(true_pos, len(pred_set), len(ref_set))
                    # weigh the F1's by how much they will contribute to overall f1.
                    weight = len(ref_set) / ref_size
                    f1 = f1*weight
                    graph.add_edge(pred_k, ref_k, weight=-f1)
                else:
                    graph.add_edge(pred_k, ref_k, weight=0)

        # 3. Find pairs of nodes that maximize the weighted F1 (or minimize inverse)
        matches = minimum_weight_full_matching(graph, pred_nodes)

        # 4. convert each sample into lemma_form, where lemma is a class_id
        #    for the form from the set keys in the cluster dicts.
        # refParadigmId_word for all words in ref
        flat_refs = [f"{k}_{w}" for k, words in self.reference_dict.items() for w in words]
        flat_preds = []
        for pred_node, ref_node in matches.items():
            if pred_node in pred_nodes:
                # refParadigmId_word for all aligned words in preds
                flat_preds.extend([f"{ref_node}_{w}" for w in pred_dict[pred_node]])
                # Remove from pred_dict, so we can loop over the leftovers
                pred_dict.pop(pred_node)

        # Assign the unmatched predictions a label from the pred set
        for k, words in pred_dict.items():
            flat_preds.extend([f"{k}_{w}" for w in words])

        flat_preds = set(flat_preds)
        flat_refs = set(flat_refs)
        # 5. now compute F1 between the labeled preds, and labeled refs
        true_pos = len(flat_refs & flat_preds)

        prec, rec, f1 = self._get_metrics(true_pos, len(flat_preds), len(flat_refs))

        return {
            "precision": prec,
            "recall": rec,
            "f1": f1
        }


def eval(pred_fn: str, ref_fn: str):
    evaluator = Evaluator(ref_dict={f"ref_{i}": set(r.words) for i, r in enumerate(iter_paradigms(ref_fn))})
    eval_dict = evaluator.score({f"pred_{i}": set(p.words) for i, p in enumerate(iter_paradigms(pred_fn))})

    for k, v in eval_dict.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate morphological paradigm clustering.')
    parser.add_argument('--reference', help='The ground truth file')
    parser.add_argument('--prediction', help='The prediction file')
    args = parser.parse_args()

    eval(args.prediction, args.reference)
