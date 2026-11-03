# HOLO-GURU Classroom Teaching Foundation

This version intentionally excludes the 3D teacher/avatar layer. It adds the backend foundation for classroom teaching behavior while preserving the existing local LLM, TTS/STT, Weaviate RAG, visual aids, and ReAct pipeline.

## Added foundation

- Persistent classroom state in SQLite (`classroom_state.db`)
- Timetable-based subject/staff switching without reloading the LLM
- Staff teaching-style profiles
- Subject → Unit → Lesson → Topic curriculum state
- Whole-class mastery gate: only PRESENT students count; absent/non-response is not treated as failure
- Adaptive reteaching with rotating methods instead of repeating the same explanation
- Visitor pause → checkpoint → student recall → short recap/resume behavior
- Individual student break behavior that never pauses the class
- "Excuse me" → brief acknowledgement only
- Whole-class break pause/checkpoint/resume
- Temporal distraction handling before gentle re-engagement
- Caring wellbeing reaction without medical/emotion diagnosis
- Natural real-teacher prompt: short explanation, real-world example, visual when useful, discussion/demonstration/storytelling when relevant
- Uploaded-book-only grounding by default; Wikipedia/web fallback is disabled unless explicitly enabled

## Main API routes

`POST /api/classroom/configure` — timetable, curriculum, staff styles, class roster

`GET /api/classroom/state?session_id=...` — current classroom + mastery + period state

`POST /api/classroom/sync-period` — switch to the period that is active now

`POST /api/classroom/start-topic` — explicitly start a curriculum topic

`POST /api/classroom/attendance` — set present/absent students

`POST /api/classroom/mastery` — record understanding for one present student

`POST /api/classroom/advance` — move to next topic only when mastery gate passes (unless teacher override is explicit)

`POST /api/classroom/event` — visitor/break/distraction/wellbeing classroom events

## Classroom event names

- `visitor_enter`
- `visitor_leave`
- `student_recall_response` with `{ "correct": true|false }`
- `student_break_leave` / `student_break_return` with `{ "student_id": "..." }`
- `student_excuse_me`
- `class_break_start` / `class_break_end`
- `distraction_signal` with `{ "student_id": "...", "detected": true|false }`
- `wellbeing_observation`
- `student_says_fine`

## Example configuration

```json
{
  "session_id": "AI-DS-3A",
  "class_students": ["S001", "S002", "S003"],
  "timetable": [
    {
      "days": ["Monday", "Wednesday"],
      "start": "09:00",
      "end": "09:50",
      "subject": "Physics",
      "staff_id": "staff_physics"
    }
  ],
  "staff_styles": {
    "staff_physics": {
      "name": "example_and_board",
      "preferred_methods": ["real_world_example", "visual_or_diagram", "step_by_step"]
    }
  },
  "curriculum": {
    "Physics": {
      "units": [
        {
          "name": "Electricity",
          "lessons": [
            {
              "name": "Electrical Basics",
              "topics": ["Electric Current", "Voltage", "Ohm's Law", "Electrical Power"]
            }
          ]
        }
      ]
    }
  }
}
```

## Accuracy policy

`RAG_ALLOW_WEB_FALLBACK=false` is the default. In classroom mode, if the uploaded books do not support an answer, HOLO-GURU should say the topic is not covered instead of silently mixing web facts into textbook teaching. Set the environment variable to `true` only when external web knowledge is explicitly desired.
