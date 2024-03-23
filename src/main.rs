use std::env;

mod server; // Import the server module

fn main() {
    // Initialize logging (optional)
    // ...

    // Get server address from environment variable or use default
    let address = env::var("SERVER_ADDRESS").unwrap_or("127.0.0.1:8080".to_string());

    // Create and start the server
    let mut server = server::Server::new(address);
    if let Err(e) = server.run() {
        println!("Error running server: {}", e);
    }
}
