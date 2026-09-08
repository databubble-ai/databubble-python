# databubble/model.py
"""
ModelClient — db.model.* : portable JSON model cards for elasticity/driver's
fitted regressions.

    card = db.model.export(result)          # result: a fitted JourneyResult
    card.save("price_model.json")           # reload anywhere, anytime
    preds = db.model.predict(card, new_df)  # score independent of the session

Cards never pickle a fitted statsmodels object — coefficients plus a
replayable FeatureRecipe (dummy encoding, log transforms, interactions) — so
scoring never needs the platform again once you have the card.
"""

from __future__ import annotations

from typing import Any
from databubble.models import ModelCardResult, PredictionResult, ComparisonResult, DriftResult, SkillResult
from databubble._scoring_common import extract_ref, artifact_dict, df_to_records
from databubble.exceptions import SDKUsageError


class ModelClient:
    def __init__(self, http_client):
        self._http = http_client

    def export(self, source: Any, format: str = "json") -> ModelCardResult:
        """
        Export the fitted model from an elasticity/driver JourneyResult as a
        portable JSON card.

        Args:
            source: a JourneyResult from db.journeys.elasticity()/driver(),
                    or the raw model_export_ref dict
                    (result.raw["result"]["model_export_ref"]).
            format: reserved for future formats; only "json" today.

        Returns:
            ModelCardResult — card.coefficients, card.save(path), card.raw.

        Example:
            result = db.journeys.elasticity(df, price_col="price", sales_col="sales")
            card = db.model.export(result)
            card.save("elasticity_price.json")

        Tier: same tier gate as the journey that produced the model.
        """
        ref = extract_ref(source, "model_export_ref", "db.model.export")
        response = self._http.post_json(
            "/v1/model/export", {"model_export_ref": ref, "format": format}
        )
        kind = "bundle" if response.get("kind") == "bundle" else "card"
        outcome = response.get("outcome") or (response.get("cards") or [{}])[0].get("outcome", "")
        return ModelCardResult(
            outcome=outcome, kind=kind, model_family=response.get("model_family"), raw=response,
        )

    def predict(self, card: Any, rows) -> PredictionResult:
        """
        Score new rows against a card — no session, no re-fitting.

        Args:
            card: a ModelCardResult (from .export()) or a dict — e.g. reloaded
                  with json.load() from a card saved earlier.
            rows: a pd.DataFrame containing the card's recipe.raw_inputs as columns.

        Returns:
            PredictionResult — preds.predictions (DataFrame), preds.raw.

        Example:
            preds = db.model.predict(card, new_prices_df)
            preds.predictions["prediction"]
        """
        card_dict = artifact_dict(card, "db.model.predict", "card")
        payload = {"card": card_dict, "rows": df_to_records(rows, "db.model.predict")}
        response = self._http.post_json("/v1/model/predict", payload)
        return PredictionResult(
            outcome=response.get("outcome", ""),
            n_scored=response.get("n_scored", 0),
            log_back_transformed=response.get("log_back_transformed", False),
            warnings=response.get("warnings", []),
            raw=response,
        )

    def compare(self, cards: list, mode: str = "ic") -> ComparisonResult:
        """
        Compare 2+ cards by information criteria ("ic", the default — ranks
        by AIC/BIC) or a nested F-test ("nested" — exactly 2 cards,
        [reduced, full]).

        Args:
            cards: list of ModelCardResult or dicts — individual cards, not bundles.
            mode:  "ic" or "nested".

        Example:
            cmp = db.model.compare([card_a, card_b, card_c])
            cmp.candidates   # DataFrame ranked by AIC
            cmp.best_label
        """
        card_dicts = [artifact_dict(c, "db.model.compare", "cards") for c in cards]
        response = self._http.post_json("/v1/model/compare", {"cards": card_dicts, "mode": mode})
        return ComparisonResult(mode=mode, outcome=response.get("outcome", ""), raw=response)

    def interval_calibration(
        self,
        card: Any,
        rows,
        actuals: list,
        interval_kind: str = "prediction",
        nominal: float = 0.95,
        wilson_conf: float = 0.95,
    ) -> SkillResult:
        """
        Backtest a card's prediction/confidence intervals against a genuine
        holdout — coverage, calibration curve, and whether the interval
        widths are honest. Same arg shape as .predict()/.drift() (card + rows,
        stateless) — the one model_export.py route this client didn't wrap.

        Args:
            card:           a ModelCardResult (from .export()) or a dict.
            rows:           a pd.DataFrame the card was NOT fitted on — a real
                             holdout, or coverage will look optimistic. This
                             endpoint cannot verify that itself.
            actuals:        list of true outcome values, same length as rows,
                             same row order.
            interval_kind:  "prediction" (pi_lower/pi_upper) or "mean_response"
                             (ci_lower/ci_upper).
            nominal:        Target interval coverage, e.g. 0.95 for a 95% interval.
            wilson_conf:    Confidence level for the Wilson interval around the
                             observed coverage rate.

        Returns:
            SkillResult — same shape as db.skills.*, since the platform route
            wraps the same interval_calibration_skill() that builds one.

        Example:
            card = db.model.export(result)
            holdout = new_df.iloc[:50]
            backtest = db.model.interval_calibration(card, holdout, holdout["sales"].tolist())
            print(backtest.summary)
        """
        from databubble.skills import _parse_skill_result

        card_dict = artifact_dict(card, "db.model.interval_calibration", "card")
        rows_records = df_to_records(rows, "db.model.interval_calibration")
        if len(actuals) != len(rows_records):
            raise SDKUsageError(
                f"db.model.interval_calibration(): actuals ({len(actuals)}) must have "
                f"the same length as rows ({len(rows_records)})."
            )
        payload = {
            "card": card_dict, "rows": rows_records, "actuals": actuals,
            "interval_kind": interval_kind, "nominal": nominal, "wilson_conf": wilson_conf,
        }
        response = self._http.post_json("/v1/model/interval-calibration", payload)
        return _parse_skill_result(response, http=self._http)

    def drift(self, card: Any, rows) -> DriftResult:
        """
        Check whether new data has drifted from the card's training-time
        feature snapshot (mean/std/skewness per continuous feature,
        proportion per categorical).

        Example:
            d = db.model.drift(card, latest_batch_df)
            if d.drift_detected:
                print(d.interpretation)
        """
        card_dict = artifact_dict(card, "db.model.drift", "card")
        payload = {"card": card_dict, "rows": df_to_records(rows, "db.model.drift")}
        response = self._http.post_json("/v1/model/drift", payload)
        return DriftResult(
            applicable=response.get("applicable", False),
            drift_detected=response.get("drift_detected", False),
            n_features_checked=response.get("n_features_checked", 0),
            n_features_drifted=response.get("n_features_drifted", 0),
            interpretation=response.get("interpretation", ""),
            raw=response,
        )
