import SwiftUI

@main
struct BatcomputerApp: App {
    @StateObject private var api = APIClient()
    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(api)
        }
    }
}