use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::fs::File;

struct EmbeddingRecord {
    id: i32,
    embedding: Vec<f32>,
}

pub fn get_number_of_users(user_embeddings_uri : &String)-> usize {
    let file = File::open(user_embeddings_uri).expect("Failed to open file");
    let reader = SerializedFileReader::new(file).expect("Failed to create reader");
    let num_rows = reader.metadata().file_metadata().num_rows();
    num_rows as usize
}

pub fn read_movie_embeddings(movie_embeddings_uri : &String) -> (Vec<f32>, usize, usize) {
    _read_embeddings(&movie_embeddings_uri)
}


pub fn read_user_embeddings(user_embeddings_uri : &String) -> (Vec<f32>, usize, usize) {
    _read_embeddings(&user_embeddings_uri)
}

pub fn get_user_embeddings(ids: &[i32], embeddings_catalog: &[f32], embed_len: usize) -> Vec<f32> {
    let n_users = ids.len();
    let mut embeddings_1d = vec![0.0; n_users * embed_len];
    for i in 0 .. n_users {
        let idx = (ids[i] -1) as usize;
        let i00 = idx * embed_len;
        let i01 = i00 + embed_len;
        let i10 = i * embed_len;
        let i11 = i10 + embed_len;
        embeddings_1d[i10 .. i11].copy_from_slice(&embeddings_catalog[i00..i01]);
    }

    embeddings_1d
}

fn _read_embeddings(embeddings_uri: &String) -> (Vec<f32>, usize, usize) {

    let file = File::open(embeddings_uri).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();

    let mut records: Vec<EmbeddingRecord> = Vec::new();

    // traverses all row groups automatically
    // The `None` projection means we read all columns.
    let row_iter = reader.get_row_iter(None).expect("Failed to get row iter");

    for record_result in row_iter {
        let row = record_result.expect("Failed to read row");

        // Assuming 'id' is column 0 and 'embedding' is column 1.
        // Adjust these indices if your schema is ordered differently.
        let id = row.get_int(0).expect("Failed to read id");

        // Extract the PyArrow list into a Rust Vec<f32>
        let list_data = row.get_list(1).expect("Failed to read embedding list");
        let embedding: Vec<f32> = list_data
            .elements()
            .iter()
            .map(|field| match field {
                // Pattern match to extract the f32 from the Field::Float variant
                parquet::record::Field::Float(f) => *f,
                // Add a fallback panic to catch schema mismatches
                // (e.g., if Parquet saved them as Doubles (f64) instead of Floats)
                _ => panic!("Expected Float field in list, but found: {:?}", field),
            })
            .collect();

        records.push(EmbeddingRecord { id, embedding });
    }

    // Sort the records by ID (ascending)
    records.sort_by_key(|r| r.id);

    // Determine the matrix shape dynamically
    let num_embeddings = records.len();
    let embed_len = records.first().map(|r| r.embedding.len()).unwrap_or(0);

    // Stack vertically into a single flat array
    // Pre-allocating capacity avoids expensive memory reallocations
    let mut stacked_embeddings: Vec<f32> = Vec::with_capacity(num_embeddings * embed_len);
    for record in records {
        stacked_embeddings.extend(record.embedding);
    }

    (stacked_embeddings, num_embeddings, embed_len)

}
