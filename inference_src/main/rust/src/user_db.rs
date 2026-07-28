use std::{path::{Path}};
use crate::pb::UserRequest;
use memmap2::Mmap;
use std::fs::File;
use std::io::Result;
use tonic::Request;

pub struct UserDb {
    mmap: Mmap,
    record_size: usize,
}

impl UserDb {
    /// Opens and memory-maps the binary users database.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;

        // Safety: Mmap is safe as long as no other process truncates
        // the file while our Rust program is running.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            mmap,
            record_size: 9, // 1 byte (c) + 4 bytes (I) + 4 bytes (I)
        })
    }

    /// Fetches a UserRequest in O(1) time.
    /// Cache misses are handled transparently by the OS Page Cache.
    pub fn get_request(&self, user_id: i64) -> Option<Request<UserRequest>> {
        if user_id == 0 {
            return None;
        }

        // Calculate offset: file_index = user_id - 1
        let index = (user_id as usize) - 1;
        let offset = index * self.record_size;

        // Boundary check
        if offset + self.record_size > self.mmap.len() {
            return None;
        }

        // Slice the 17 bytes for this specific user
        let chunk = &self.mmap[offset..offset + self.record_size];

        // Parse Gender (1 byte)
        let gender = match chunk[0] {
            b'M' => "M".to_string(),
            b'F' => "F".to_string(),
            _ => "U".to_string(), // Unknown fallback
        };

        // Parse age (4 bytes, Little Endian)
        let age = i64::from_le_bytes(chunk[1..5].try_into().unwrap());

        // Parse occupation (4 bytes, Little Endian)
        let occupation = i64::from_le_bytes(chunk[5..9].try_into().unwrap());

        //let timestamp2003: i64 = 1044144000;
        let timestamp2001 : i64 = 956703932;
        let timestamp = timestamp2001;

        Some(Request::new(UserRequest {
            user_id,
            gender,
            occupation,
            age,
            timestamp,
        }))
    }
}