import Foundation
import UIKit

struct ChatRequest: Codable {
    let message: String
    let include_vision: Bool
    let use_reasoning: Bool
    let show_reasoning: Bool
    let reasoning_preset: String
    let image_base64: String?
}

struct ChatResponse: Codable { let response: String }

struct PromptTemplate: Codable, Identifiable {
    var id: String? = nil
    var name: String
    var content: String
}

struct ExecuteSQLRequest: Codable {
    let sql: String
    let params: [String]?
}

struct SQLResult: Codable {
    let columns: [String]
    let rows: [[String: String]]
    let status: String?
}

final class APIClient: ObservableObject {
    @Published var baseURL: String = "http://127.0.0.1:8000" // Set LAN IP here

    private func makeURL(_ path: String) -> URL { URL(string: baseURL + path)! }

    func chat(message: String, preset: String, showReasoning: Bool, image: UIImage?) async throws -> String {
        let imageData = image?.jpegData(compressionQuality: 0.8)?.base64EncodedString()
        let body = ChatRequest(
            message: message,
            include_vision: image != nil,
            use_reasoning: true,
            show_reasoning: showReasoning,
            reasoning_preset: preset,
            image_base64: imageData
        )
        var req = URLRequest(url: makeURL("/chat"))
        req.httpMethod = "POST"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(ChatResponse.self, from: data).response
    }

    func listPrompts() async throws -> [PromptTemplate] {
        let (data, _) = try await URLSession.shared.data(from: makeURL("/prompts"))
        return try JSONDecoder().decode([PromptTemplate].self, from: data)
    }

    func createPrompt(_ tpl: PromptTemplate) async throws -> PromptTemplate {
        var req = URLRequest(url: makeURL("/prompts"))
        req.httpMethod = "POST"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(tpl)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(PromptTemplate.self, from: data)
    }

    func updatePrompt(id: String, tpl: PromptTemplate) async throws -> PromptTemplate {
        var req = URLRequest(url: makeURL("/prompts/\(id)"))
        req.httpMethod = "PUT"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(tpl)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(PromptTemplate.self, from: data)
    }

    func deletePrompt(id: String) async throws {
        var req = URLRequest(url: makeURL("/prompts/\(id)"))
        req.httpMethod = "DELETE"
        _ = try await URLSession.shared.data(for: req)
    }

    func listDBs() async throws -> [String] {
        let (data, _) = try await URLSession.shared.data(from: makeURL("/dbs"))
        return try JSONDecoder().decode([String].self, from: data)
    }

    func createDB(name: String) async throws {
        struct CreateDB: Codable { let name: String }
        var req = URLRequest(url: makeURL("/dbs"))
        req.httpMethod = "POST"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(CreateDB(name: name))
        _ = try await URLSession.shared.data(for: req)
    }

    func deleteDB(name: String) async throws {
        var req = URLRequest(url: makeURL("/dbs/\(name)"))
        req.httpMethod = "DELETE"
        _ = try await URLSession.shared.data(for: req)
    }

    func listTables(db: String) async throws -> [String] {
        let (data, _) = try await URLSession.shared.data(from: makeURL("/dbs/\(db)/tables"))
        struct R: Codable { let tables: [String] }
        return try JSONDecoder().decode(R.self, from: data).tables
    }

    func executeSQL(db: String, sql: String) async throws -> SQLResult {
        var req = URLRequest(url: makeURL("/dbs/\(db)/execute"))
        req.httpMethod = "POST"
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(ExecuteSQLRequest(sql: sql, params: nil))
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(SQLResult.self, from: data)
    }
}