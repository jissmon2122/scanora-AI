"""
Scanora AI — Multi-View (CC & MLO) and Examination-Level Diagnostic Aggregator.

Clinically aggregates:
- Left Breast Views: L-CC (Craniocaudal) + L-MLO (Mediolateral Oblique)
- Right Breast Views: R-CC + R-MLO
Computes breast-level and patient-level risk according to standardized ACR guidelines.
"""

from typing import Dict, List, Any, Optional


class MultiViewAggregator:
    @staticmethod
    def aggregate_breast_views(
        views_data: List[Dict[str, Any]], laterality: str
    ) -> Dict[str, Any]:
        """
        Aggregates views for a single breast (e.g. Right CC + Right MLO).
        Checks for view concordance (findings detected in both orthogonal projections).
        """
        if not views_data:
            return {
                "laterality": laterality,
                "status": "NO_VIEWS",
                "max_malignancy_prob": 0.0,
                "has_concordance": False,
            }

        views_present = [v.get("view", "UNKNOWN") for v in views_data]
        is_complete_two_view = "CC" in views_present and "MLO" in views_present

        # Extract calibrated probabilities
        probs = [v.get("calibrated_malignancy_probability", 0.0) for v in views_data]
        max_prob = max(probs)

        # Check for lesion concordance between views
        all_findings = []
        for v in views_data:
            all_findings.extend(v.get("findings", []))

        # Check if suspicious lesions appear in both orthogonal views
        concordant_types = set()
        types_by_view = {
            v.get("view", f"v_{i}"): [f.get("type") for f in v.get("findings", [])]
            for i, v in enumerate(views_data)
        }

        if len(types_by_view) >= 2:
            view_keys = list(types_by_view.keys())
            types_view1 = set(types_by_view[view_keys[0]])
            types_view2 = set(types_by_view[view_keys[1]])
            concordant_types = types_view1.intersection(types_view2)

        has_concordance = len(concordant_types) > 0

        # Clinical concordance weighting: a lesion verified in both CC and MLO views
        # has higher positive predictive value than a unilateral single-view finding (which may be tissue summation)
        if has_concordance and max_prob > 0.15:
            calibrated_breast_prob = min(0.99, max_prob * 1.15)
        elif not is_complete_two_view and max_prob > 0.30:
            # Single view only: indicate incomplete study caution
            calibrated_breast_prob = max_prob
        else:
            calibrated_breast_prob = max_prob

        return {
            "laterality": laterality,
            "views_evaluated": views_present,
            "is_complete_study": is_complete_two_view,
            "breast_malignancy_prob": round(calibrated_breast_prob, 4),
            "concordant_lesions": list(concordant_types),
            "has_orthogonal_concordance": has_concordance,
            "total_findings": len(all_findings),
        }

    @classmethod
    def aggregate_examination(
        cls, left_breast_result: Dict[str, Any], right_breast_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Computes overall study assessment.
        Overall case category is driven by the highest risk breast.
        """
        left_prob = left_breast_result.get("breast_malignancy_prob", 0.0)
        right_prob = right_breast_result.get("breast_malignancy_prob", 0.0)

        case_max_prob = max(left_prob, right_prob)
        higher_risk_side = (
            "Bilateral"
            if abs(left_prob - right_prob) < 0.05 and case_max_prob > 0.2
            else ("Left Breast" if left_prob > right_prob else "Right Breast")
        )

        # Standardized research assessment guidance
        if case_max_prob < 0.02:
            guideline = "Negative for suspicious malignancy. Routine annual screening recommended."
            research_assessment = "Low Risk (Negative / Benign Profile)"
        elif case_max_prob < 0.08:
            guideline = "Benign or probably benign imaging findings. 6-month short-interval surveillance or routine follow-up."
            research_assessment = "Intermediate Low Risk"
        elif case_max_prob < 0.50:
            guideline = "Suspicious findings noted. Targeted diagnostic ultrasound and clinical correlation recommended."
            research_assessment = "Suspicious / Moderate Risk"
        else:
            guideline = "High probability of malignancy. Urgent tissue biopsy and multidisciplinary oncology review indicated."
            research_assessment = "High Risk / Highly Suspicious"

        return {
            "examination_risk_score": round(case_max_prob * 100.0, 2),
            "primary_region_of_concern": higher_risk_side,
            "research_assessment_category": research_assessment,
            "clinical_management_recommendation": guideline,
            "left_breast": left_breast_result,
            "right_breast": right_breast_result,
        }
