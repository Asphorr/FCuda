use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use std::thread;

pub struct Server {
    address: String,
}

impl Server {
    pub fn new(address: String) -> Self {
        Server { address }
    }

    pub fn run(&mut self) -> Result<(), std::io::Error> {
        let listener = TcpListener::bind(&self.address)?;
        println!("Server listening on {}", self.address);

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    thread::spawn(move || {
                        self.handle_client(stream);
                    });
                }
                Err(e) => {
                    println!("Error accepting connection: {}", e);
                }
            }
        }

        Ok(())
    }

    fn handle_client(&mut self, mut stream: TcpStream) {
        let mut buffer = [0; 1024];
        
        loop {
            match stream.read(&mut buffer) {
                Ok(size) => {
                    if size == 0 {
                        break; // Client disconnected
                    }
                    // Process the request (replace with your actual logic)
                    let response = format!("Hello from Rust server! You sent: {}", 
                                          String::from_utf8_lossy(&buffer[..size]));
                    stream.write(response.as_bytes()).unwrap();
                }
                Err(e) => {
                    println!("Error reading from client: {}", e);
                    break;
                }
            }
        }
    }
}
