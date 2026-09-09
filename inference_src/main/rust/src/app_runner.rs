use tonic::transport::Server;
// Import the tonic-health reporter
use tonic_health::server::health_reporter;
use crate::app_config::AppConfig;
use crate::orchestrator::Orchestrator;
use crate::pb::recommender_service_server::RecommenderServiceServer;
use crate::util::{check_path};

pub struct AppRunner {
    config: AppConfig,
}

impl AppRunner {
    pub fn new(config: AppConfig) -> Self {
        Self { config }
    }

    pub async fn run<F>(self, shutdown_signal: F,
        // Add an optional channel sender to report the bound address
        addr_sender: Option<tokio::sync::oneshot::Sender<std::net::SocketAddr>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where F: std::future::Future<Output = ()> + Send + 'static, {

        self.validate_paths()?;
        println!("Starting Recommender Service on {}", self.config.server_addr);

        // Initialize the gRPC Health Reporter
        let (health_reporter, health_service) = health_reporter();

        // Mark your specific service as NotServing during initialization
        // The string must match the exact package and service name in your .proto file
        let service_name = "recommender.RecommenderService";
        health_reporter
            .set_service_status(service_name, tonic_health::ServingStatus::NotServing)
            .await;

        // ... [Load JSON Hyperparameters] ...
        //let file = File::open(&self.config.params_json_path)?;
        //let reader = BufReader::new(file);
        //let dict: HashMap<String, Value> = serde_json::from_reader(reader)?;

        let ratings_uris_refs: Vec<&str> = self.config.ratings_uris.iter().map(|s| s.as_str()).collect();

        println!("Building Orchestrator (loading embeddings and user history)...");

        // The heavy lifting: connect to TFS and load embeddings
        let orchestrator = Orchestrator::new(
            self.config.query_uri,
            self.config.ranker_uri,
            self.config.query_metadata_uri,
            self.config.ranker_metadata_uri,
            &self.config.movie_embeddings_path,
            ratings_uris_refs,
            self.config.ranker_n_local_devices,
            self.config.top_k,
            self.config.persisted_index_path,
            self.config.user_db_path
        ).await?;

        let listener = tokio::net::TcpListener::bind(self.config.server_addr).await?;
        let actual_addr = listener.local_addr()?;
        println!("Server listening on: {}", actual_addr);

        //transmite the actual addr back to caller
        if let Some(tx) = addr_sender {
            let _ = tx.send(actual_addr);
        }

        // Initialization complete. Update  Health Reporter
        health_reporter
            .set_service_status(service_name, tonic_health::ServingStatus::Serving)
            .await;

        println!("Service State updated to Ready. Listening for gRPC traffic...");

        // set to 256 MB.  this can handle a RankedMovieResponse for 700_000 user_ids in a UsersRequest
        const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;

        // Start the server with BOTH the health service and your recommender service
        Server::builder()
            .add_service(health_service) // Injects the standard grpc.health.v1.Health service
            .add_service(
                RecommenderServiceServer::new(orchestrator)
                    .max_decoding_message_size(MAX_MESSAGE_SIZE)
                    .max_encoding_message_size(MAX_MESSAGE_SIZE))
            .serve_with_incoming_shutdown(
                tokio_stream::wrappers::TcpListenerStream::new(listener),
                shutdown_signal,
            )
            .await?;

        Ok(())
    }

    fn validate_paths(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

        println!("Running from directory: {:?}", std::env::current_dir()?);

        check_path(&self.config.params_json_path, "Hyperparameters JSON")?;
        check_path(&self.config.movie_embeddings_path, "Movie Embeddings Binary")?;

        // 2. Iterate through the vector of dynamic ratings paths
        for (index, uri) in self.config.ratings_uris.iter().enumerate() {
            check_path(uri, &format!("Ratings File #{}", index + 1))?;
        }

        Ok(())
    }
}