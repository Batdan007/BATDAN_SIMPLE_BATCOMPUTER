import SwiftUI
import AVFoundation
import UIKit

final class ChatViewModel: ObservableObject {
    @Published var messages: [Message] = []
    @Published var input: String = ""
    @Published var preset: String = "default"
    @Published var showReasoning: Bool = true
    @Published var pickedImage: UIImage? = nil
    let synthesizer = AVSpeechSynthesizer()

    struct Message: Identifiable { let id = UUID(); let role: String; let content: String }

    func speak(_ text: String) {
        synthesizer.speak(AVSpeechUtterance(string: text))
    }
}

struct RootView: View {
    var body: some View {
        TabView {
            ChatView()
                .tabItem { Label("Chat", systemImage: "message") }
            PromptsView()
                .tabItem { Label("Prompts", systemImage: "text.badge.plus") }
            DatabasesView()
                .tabItem { Label("Databases", systemImage: "table") }
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gear") }
        }
    }
}

struct ChatView: View {
    @EnvironmentObject var api: APIClient
    @StateObject var vm = ChatViewModel()
    @State private var isPicking = false

    var body: some View {
        VStack(spacing: 12) {
            Picker("Preset", selection: $vm.preset) {
                Text("Default").tag("default")
                Text("Creative").tag("creative")
                Text("Analysis").tag("analysis")
                Text("Problem Solving").tag("problem_solving")
            }.pickerStyle(.segmented)

            Toggle("Show reasoning", isOn: $vm.showReasoning)

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(vm.messages) { m in
                        HStack {
                            if m.role == "user" { Spacer() }
                            Text(m.content).padding(10)
                                .background(m.role == "user" ? Color.blue.opacity(0.15) : Color.gray.opacity(0.15))
                                .cornerRadius(8)
                            if m.role == "assistant" { Spacer() }
                        }
                    }
                }
            }

            if let img = vm.pickedImage {
                Image(uiImage: img).resizable().scaledToFit().frame(height: 150).cornerRadius(8)
            }

            HStack {
                Button(action: { isPicking = true }) { Image(systemName: "camera") }
                TextField("Message", text: $vm.input).textFieldStyle(.roundedBorder)
                Button("Send") { Task { await send() } }
            }
        }
        .padding()
        .sheet(isPresented: $isPicking) {
            ImagePicker(image: $vm.pickedImage)
        }
    }

    func send() async {
        guard !vm.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || vm.pickedImage != nil else { return }
        let text = vm.input
        vm.messages.append(.init(role: "user", content: text))
        vm.input = ""
        do {
            let resp = try await api.chat(message: text, preset: vm.preset, showReasoning: vm.showReasoning, image: vm.pickedImage)
            vm.messages.append(.init(role: "assistant", content: resp))
            vm.speak(resp)
            vm.pickedImage = nil
        } catch {
            vm.messages.append(.init(role: "assistant", content: "Error: \(error.localizedDescription)"))
        }
    }
}

struct PromptsView: View {
    @EnvironmentObject var api: APIClient
    @State private var templates: [PromptTemplate] = []
    @State private var name = ""
    @State private var content = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                TextField("Template name", text: $name).textFieldStyle(.roundedBorder)
                Button("Add") { Task { await add() } }
            }
            TextEditor(text: $content).frame(minHeight: 120).overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.2)))
            HStack { Spacer(); Button("Save Template") { Task { await add() } } }
            List {
                ForEach(templates) { t in
                    VStack(alignment: .leading) {
                        Text(t.name).font(.headline)
                        Text(t.content).font(.subheadline).foregroundColor(.secondary)
                    }
                    .swipeActions {
                        Button(role: .destructive) { Task { await del(t) } } label: { Label("Delete", systemImage: "trash") }
                    }
                }
            }.listStyle(.plain)
        }
        .padding()
        .task { await reload() }
    }

    func reload() async { templates = (try? await api.listPrompts()) ?? [] }

    func add() async {
        guard !name.isEmpty, !content.isEmpty else { return }
        _ = try? await api.createPrompt(.init(id: nil, name: name, content: content))
        name = ""; content = ""; await reload()
    }

    func del(_ t: PromptTemplate) async { if let id = t.id { try? await api.deletePrompt(id: id); await reload() } }
}

struct DatabasesView: View {
    @EnvironmentObject var api: APIClient
    @State private var dbs: [String] = []
    @State private var newName: String = ""
    @State private var selected: String? = nil
    @State private var tables: [String] = []
    @State private var sql: String = "SELECT name FROM sqlite_master WHERE type='table'"
    @State private var result: SQLResult? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                TextField("New DB name", text: $newName).textFieldStyle(.roundedBorder)
                Button("Create") { Task { try? await api.createDB(name: newName); newName = ""; await loadDBs() } }
            }
            List(selection: $selected) {
                ForEach(dbs, id: \.self) { db in
                    Text(db).onTapGesture { Task { selected = db; await loadTables() } }
                        .swipeActions { Button(role: .destructive) { Task { try? await api.deleteDB(name: db); await loadDBs() } } label: { Label("Delete", systemImage: "trash") } }
                }
            }.listStyle(.plain)

            if let db = selected {
                Text("Selected: \(db)").font(.headline)
                ScrollView(.horizontal) { Text(sql).font(.system(.body, design: .monospaced)) }
                HStack {
                    Button("Run SQL") { Task { try? await runSQL() } }
                    Button("Refresh Tables") { Task { await loadTables() } }
                }
                if let r = result {
                    ScrollView {
                        VStack(alignment: .leading) {
                            if !r.columns.isEmpty {
                                Text(r.columns.joined(separator: " | ")).bold()
                            }
                            ForEach(Array(r.rows.enumerated()), id: \.offset) { _, row in
                                Text(r.columns.map { row[$0] ?? "" }.joined(separator: " | "))
                            }
                        }
                    }
                    .frame(maxHeight: 200)
                }
            }
        }
        .padding()
        .task { await loadDBs() }
    }

    func loadDBs() async { dbs = (try? await api.listDBs()) ?? [] }
    func loadTables() async { if let db = selected { tables = (try? await api.listTables(db: db)) ?? [] } }
    func runSQL() async { if let db = selected { result = try? await api.executeSQL(db: db, sql: sql) } }
}

struct SettingsView: View {
    @EnvironmentObject var api: APIClient
    var body: some View {
        Form {
            Section(header: Text("Server")) {
                TextField("Base URL", text: $api.baseURL)
                    .keyboardType(.URL)
                    .autocapitalization(.none)
            }
        }
    }
}

// MARK: - Image Picker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    func makeUIViewController(context: Context) -> some UIViewController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ uiViewController: UIViewControllerType, context: Context) {}
    func makeCoordinator() -> Coord { Coord(self) }
    final class Coord: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: ImagePicker
        init(_ p: ImagePicker) { self.parent = p }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let img = info[.originalImage] as? UIImage { parent.image = img }
            picker.dismiss(animated: true)
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { picker.dismiss(animated: true) }
    }
}