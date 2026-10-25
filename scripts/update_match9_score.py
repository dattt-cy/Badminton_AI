"""Update Match 9 rallies and analysis in database with calibrated scoreboard OCR."""
import json
import psycopg2
from psycopg2.extras import Json
from ai_classifier.analytics import build_match_analysis, calibrate_bwf_scoreboard_roi, extract_score_events

def main():
    conn = psycopg2.connect(
        "postgresql://postgres:aktKLxEeSWE4cxTA@db.vdngbhajsksqqluvdfrf.supabase.co:5432/postgres?sslmode=require"
    )
    cur = conn.cursor()

    cur.execute("SELECT id, raw_result FROM ai_analyses WHERE match_id = 9 ORDER BY id DESC LIMIT 1;")
    row = cur.fetchone()
    if not row:
        print("Analysis not found")
        return

    analysis_id, raw_res = row
    predictions = raw_res["predictions"]
    print(f"Updating analysis_id={analysis_id} with {len(predictions)} predictions...")

    # Run calibrated OCR
    video_path = "work_dirs/match13_5min_clip.mp4"
    roi = calibrate_bwf_scoreboard_roi(720, 1280)
    print(f"Running extract_score_events with ROI {roi}...")
    ocr_res = extract_score_events(video_path, roi, sample_interval_frames=25)
    print("OCR initial:", ocr_res.get("initial_score"))
    print("OCR observed:", ocr_res.get("observed_score"))
    print("OCR events count:", len(ocr_res.get("events", [])))

    # Re-build match analysis
    tactics = build_match_analysis(
        predictions,
        player_names={"upper": "Kento Momota", "lower": "Viktor Axelsen"},
        fps=25.0,
        total_frames=7557,
        scoreboard_ocr=ocr_res
    )

    scoreboard = tactics["scoreboard"]
    rallies_data = scoreboard["rallies"]

    print("\n--- UPDATED RALLIES ---")
    for r in rallies_data:
        print(f"Rally {r['rally_id']}: {r['start_time_seconds']}s - {r['end_time_seconds']}s | winner: {r['winner']} ({r['winner_name']}) | score: {r['score_text']} | reason: {r['win_reason']}")

    # Update rallies in DB
    for r in rallies_data:
        cur.execute("""
            UPDATE rallies
            SET winner_side = %s,
                win_reason = %s,
                score_upper = %s,
                score_lower = %s,
                score_text = %s,
                is_complete = %s
            WHERE analysis_id = %s AND rally_number = %s;
        """, (
            str(r.get("winner", "UNKNOWN")).upper(),
            r.get("win_reason"),
            r.get("score_upper", 0),
            r.get("score_lower", 0),
            r.get("score_text"),
            bool(r.get("is_complete", True)),
            analysis_id,
            r.get("rally_id")
        ))

    # Update ai_analyses record
    raw_res["scoreboard"] = scoreboard
    raw_res["scoreboard_ocr"] = ocr_res
    if "summary" in raw_res and isinstance(raw_res["summary"], dict):
        raw_res["summary"]["estimated_score"] = scoreboard["estimated_score"]

    cur.execute("""
        UPDATE ai_analyses
        SET scoreboard_data = %s,
            scoreboard_ocr_data = %s,
            raw_result = %s,
            summary_data = %s
        WHERE id = %s;
    """, (
        Json(scoreboard),
        Json(ocr_res),
        Json(raw_res),
        Json(raw_res.get("summary", {})),
        analysis_id
    ))

    conn.commit()
    print("\n[SUCCESS] Successfully updated rallies and ai_analyses for Match 9 in database!")
    conn.close()

if __name__ == "__main__":
    main()

