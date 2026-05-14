# Moodle Integration Plan (For Teammate)

This document covers the Moodle-side work only.

## A) JWT for iframe user context

1) Add admin setting for JWT secret
- Create settings.php in the block plugin
- Add a config setting named shared_jwt_secret

2) Generate JWT in the block render
- In block_intelligent_tutor.php, pull Moodle user + course context
- Build claims:
  - userid, username, email, firstname, lastname, fullname
  - courseid, courseshortname, coursefullname
  - role (student | coordinator | teacher | instructor | editingteacher)
  - iat, exp, iss, aud
- TTL: exp = iat + 3600 (1 hour)
- Sign HS256 using shared_jwt_secret

3) Pass token to backend iframe
- iframe src should include ?token=<jwt>
- Example: https://backend.example.com/embed?token=...

## B) Moodle event observers (course creation + enrollment)

1) Register observers
- Add db/events.php with observers for:
  - course created
  - user enrolled

2) Implement observer handlers
- Add classes/observer.php (or similar)
- On event:
  - Build payload (course and/or user fields)
  - Sign a server-to-server token (separate secret recommended)
  - POST to backend:
    - POST /moodle/course-created
    - POST /moodle/user-enrolled

3) Role-based registration
- When rendering the block JWT, set role for each user:
  - student for learners
  - coordinator/teacher/instructor/editingteacher for course owners
- Backend /register will only create COORDINATOR_OF for roles above

3) Error handling
- Log failures to Moodle logs
- Do not block Moodle flow
- Optional: retry queue later

## C) Data mapping

Course created payload:
- courseid, shortname, fullname

Coordinator relationship:
- backend should create (:User)-[:COORDINATOR_OF]->(:Course)
- use /register or a new event handler to set COORDINATOR_OF

User enrolled payload:
- userid, username, email, fullname, courseid

## D) Security notes

- Use HTTPS for backend URLs
- Use separate server-to-server secret for event calls
- Keep JWTs short-lived (1 hour)
