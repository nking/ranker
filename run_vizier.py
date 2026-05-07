import os
from vizier._src.service import vizier_server

def main():
    # Fetch settings from environment variables
    port = int(os.environ.get('PORT', 8080))
    db_url = os.environ.get('VIZIER_DATABASE_URL')

    print(f"Starting Vizier Server on port {port}...")
    print(f"Connecting to database: {db_url}")

    # Initialize and serve
    # DefaultVizierServer combines the Pythia and Vizier services
    server = vizier_server.DefaultVizierServer(host='0.0.0.0', port=port, database_uri=db_url)
    server.wait_for_termination()

if __name__ == '__main__':
    main()
