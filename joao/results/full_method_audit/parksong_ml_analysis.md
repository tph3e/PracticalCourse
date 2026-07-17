# Parksong Ml Analysis

`ParkSongMLIntegration` adds ML-derived future-task predictions to ParkSong. The model component is whatever trained `PredictiveBranchingEngine` instance is supplied to `MLPredictionAdapter`; the adapter checks `is_trained`, requires `model`, extracts features, calls `predict_proba`, filters impossible activities, and emits `Prediction` objects.

Controlled comparison rows generated: 7.

ParkSong without ML still operates using explicit predictions or no predictions. With no predictions it reduces to current-task cost-based assignment. Low confidence is controlled by ParkSong's `prediction_probability_threshold`; wrong predictions are handled by integration cancellation/expiry when scheduled/executed tasks do not match.
