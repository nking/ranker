#[cfg(test)]
mod bayesian_tests {
    use std::collections::{HashMap};
    //use std::error::Error;
    use inference_engine::bayesian::{build_bayesian_catalog, calculate_user_mainstreamness, Movie, UserHistory};

    //use super::*;
    mod helper {
        // Tell Rust to literally include the code from helper.rs here
        include!("helper.rs");
    }

    fn create_movie(id: i32, title: &str, counts: [u32; 5]) -> Movie {
        Movie {
            movie_id: id,
            title: title.to_string(),
            genres: Vec::new(), // Empty for the scope of this test
            rating_counts: counts,
        }
    }

    #[test]
    fn test_bayesian_catalog_and_mainstreamness() {
        // ---------------------------------------------------------
        // 1. Build the Catalog mimicking the provided Python data
        // ---------------------------------------------------------
        let mut movies: HashMap<i32, Movie> = HashMap::new();

        movies.insert(1, create_movie(1, "loved_many",      [500,  100,  200, 3000, 8000]));
        movies.insert(2, create_movie(2, "loved_few",       [  2,    1,    0,    5,   40]));
        movies.insert(3, create_movie(3, "loved_and_hated", [4000, 1000,  500, 1000, 4000]));
        movies.insert(4, create_movie(4, "hated",           [800,  100,   50,   20,   10]));
        movies.insert(5, create_movie(5, "new_unrated",     [  0,    0,    0,    0,    0]));
        movies.insert(6, create_movie(6, "hated_few",       [ 40,    5,    0,    0,    0]));

        // ---------------------------------------------------------
        // 2. Build User Histories (Now flat slices of UserHistory)
        // ---------------------------------------------------------

        // The "Mainstream" User: Watches universally loved blockbusters and rates them highly.
        let mainstream_history = vec![
            UserHistory { movie_id: 1, rating: 5.0 }, // loved_many (S_i = 4.5136)
        ];

        // The "Niche/Tail" User: Watches obscure or poorly received items.
        // We ensure they don't heavily upvote high-S_i items to keep their M_u near/below the global mean.
        let niche_history = vec![
            UserHistory { movie_id: 2, rating: 5.0 }, // loved_few (S_i = 4.1806)
            UserHistory { movie_id: 6, rating: 4.0 }, // hated_few (S_i = 2.4445)
        ];

        // The "Hater" User: Watches universally disliked items.
        let hater_history = vec![
            UserHistory { movie_id: 4, rating: 2.0 }, // hated (S_i = 1.4176)
        ];

        // ---------------------------------------------------------
        // 3. Execute Core Functions
        // ---------------------------------------------------------
        let catalog_stats = build_bayesian_catalog(&movies);

        let mainstream_mu = calculate_user_mainstreamness(&mainstream_history, &catalog_stats);
        let niche_mu = calculate_user_mainstreamness(&niche_history, &catalog_stats);
        let hater_mu = calculate_user_mainstreamness(&hater_history, &catalog_stats);

        // ---------------------------------------------------------
        // 4. Print and Validate Results
        // ---------------------------------------------------------
        println!("===========================================");
        println!("  BAYESIAN ITEM SCORES (S_i)               ");
        println!("===========================================");

        let mut sorted_movies: Vec<_> = movies.keys().copied().collect();
        sorted_movies.sort();

        for movie_id in sorted_movies {
            let title = &movies.get(&movie_id).unwrap().title;
            let bayesian_score = catalog_stats.bayesian_scores.get(&movie_id).unwrap_or(&0.0);
            println!("Movie {:<2}: {:<16} | S_i = {:.4}", movie_id, title, bayesian_score);
        }

        println!("\n===========================================");
        println!("  USER MAINSTREAMNESS SCORES (M_u)         ");
        println!("===========================================");

        println!("User 101 (Mainstream) | M_u = {:.4}", mainstream_mu.unwrap());
        println!("User 102 (Niche/Tail) | M_u = {:.4}", niche_mu.unwrap());
        println!("User 103 (Hater)      | M_u = {:.4}", hater_mu.unwrap());

        println!("===========================================\n");

        // ---------------------------------------------------------
        // 5. Assertions for CI/CD
        // ---------------------------------------------------------
        // 'loved_many' (High count, high rating) should easily beat the global mean.  global_mean_c=3.69460487
        //let score = catalog_stats.bayesian_scores.get(&1).unwrap_or(&0.0);
        //println!("Movie 1 Bayesian Score: {:.4}", score);
        assert!(catalog_stats.bayesian_scores[&1] > catalog_stats.global_mean_c);

        // 'loved_few' raw average calculation: (2*1 + 1*2 + 0*3 + 5*4 + 40*5) / 48 = 224 / 48 = 4.666...
        // Because of its low rating count, it is aggressively pulled toward the global mean (approx 3.70).
        let loved_few_raw_avg = 224.0 / 48.0; //4.667
        assert!(catalog_stats.bayesian_scores[&2] < loved_few_raw_avg);

        assert!(mainstream_mu > niche_mu, "Mainstream M_u should exceed Niche M_u");
        // The Niche probe should maintain a solid gap above the Hater probe
        assert!(niche_mu > hater_mu, "Niche M_u should exceed Hater M_u");
    }
}