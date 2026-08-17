use std::{path::{Path}};
use crate::pb::{UserRequest, BatchUserRequest};
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
        let age = u32::from_le_bytes(chunk[1..5].try_into().unwrap());

        // Parse occupation (4 bytes, Little Endian)
        let occupation = u32::from_le_bytes(chunk[5..9].try_into().unwrap());

        // for production, would obviously use timestamp for now instead
        //let timestamp2003: i64 = 1044144000;
        let timestamp2001 : i64 = 956703932;
        let timestamp = timestamp2001;

        Some(Request::new(UserRequest {
            user_id,
            gender,
            occupation: i64::from(occupation),
            age : i64::from(age),
            timestamp,
        }))
    }

    pub fn get_batch_request(&self, user_ids: Vec<i64>) -> Option<Request<BatchUserRequest>> {
        /*
        message BatchUserRequest {
          repeated int64 user_ids = 1;
          repeated string genders = 2;
          repeated int64 occupations = 3;
          repeated int64 ages = 4;
          int32 n_users = 5;
          int64 timestamp = 6;
        }
        */
        if user_ids.is_empty() {
            return None;
        }

        // Pre-allocate vectors to avoid reallocation overhead during the loop
        let capacity : usize = user_ids.len();
        let mut valid_user_ids : Vec<i64> = Vec::with_capacity(capacity);
        let mut genders : Vec<String> = Vec::with_capacity(capacity);
        let mut occupations : Vec<i64> = Vec::with_capacity(capacity);
        let mut ages: Vec<i64>  = Vec::with_capacity(capacity);

        for user_id in user_ids {
            // Guard against 0 or negative IDs to prevent wrapping panics on the usize cast
            if user_id <= 0 {
                continue;
            }

            // Calculate offset: file_index = user_id - 1
            let index = (user_id as usize) - 1;
            let offset = index * self.record_size;

            // Boundary check
            if offset + self.record_size > self.mmap.len() {
                continue;
            }

            // Slice the bytes for this specific user
            let chunk = &self.mmap[offset..offset + self.record_size];

            // Parse Gender (1 byte)
            let gender = match chunk[0] {
                b'M' => "M".to_string(),
                b'F' => "F".to_string(),
                _ => "U".to_string(), // Unknown fallback
            };

            // Parse age (4 bytes, Little Endian)
            let age = u32::from_le_bytes(chunk[1..5].try_into().unwrap());

            // Parse occupation (4 bytes, Little Endian)
            let occupation = u32::from_le_bytes(chunk[5..9].try_into().unwrap());

            valid_user_ids.push(user_id);
            genders.push(gender);
            ages.push(i64::from(age));
            occupations.push(i64::from(occupation));
        }

        // If none of the requested IDs were valid, return None
        if valid_user_ids.is_empty() {
            return None;
        }

        let n_users = valid_user_ids.len() as i32;
        // for production, would obviously use timestamp for now instead
        //let timestamp2003: i64 = 1044144000;
        let timestamp2001 : i64 = 956703932;
        let timestamp = timestamp2001;

        Some(Request::new(BatchUserRequest {
            user_ids: valid_user_ids,
            genders,
            occupations,
            ages,
            n_users,
            timestamp: timestamp,
        }))


    }
}