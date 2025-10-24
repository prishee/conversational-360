-- ============================================================
-- VECTOR INDEXES - Will be created AFTER embeddings are generated
-- ============================================================

-- NOTE: These will fail until embeddings are generated
-- Run generate_embeddings.py first, then re-run this file

-- Create index on support tickets (only if embeddings exist)
CREATE VECTOR INDEX IF NOT EXISTS support_tickets_vector_idx
ON `conversational.support_tickets_embedded`(embedding)
OPTIONS(
  distance_type = 'COSINE',
  index_type = 'IVF',
  ivf_options = '{"num_lists": 100}'
);

-- Create index on products (only if embeddings exist)
CREATE VECTOR INDEX IF NOT EXISTS product_catalog_vector_idx
ON `conversational.product_catalog_embedded`(embedding)
OPTIONS(
  distance_type = 'COSINE',
  index_type = 'IVF',
  ivf_options = '{"num_lists": 50}'
);