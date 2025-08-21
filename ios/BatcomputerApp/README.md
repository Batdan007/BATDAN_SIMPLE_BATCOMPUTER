# Batcomputer iOS App (SwiftUI)

This is a minimal SwiftUI client for the BATCOMPUTER server.

Setup

1) Create a new Xcode iOS App (SwiftUI, iOS 16 or later) named `BatcomputerApp`.
2) Add the files from this folder into your Xcode project.
3) In `APIClient.swift`, set `baseURL` to your server, e.g. `http://192.168.1.10:8000`.
4) Run the app on your iPhone. Ensure the phone and server machine are on the same Wi‑Fi.

Capabilities

- Chat with reasoning presets; optional camera (photo upload) and voice output (Text-to-Speech).
- Manage Prompt Templates (CRUD).
- Manage SQLite Databases: create/delete DBs, list tables, execute SQL.