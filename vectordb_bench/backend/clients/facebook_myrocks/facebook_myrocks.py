# import logging
# from contextlib import contextmanager
# import json

# import mysql.connector
# import numpy as np

# from ..api import VectorDB
# from .config import FacebookMyRocksConfigDict, FacebookMyRocksIndexConfig

# log = logging.getLogger(__name__)


# class FacebookMyRocks(VectorDB):
#     def __init__(
#         self,
#         dim: int,
#         db_config: FacebookMyRocksConfigDict,
#         db_case_config: FacebookMyRocksIndexConfig,
#         collection_name: str = "vec_collection",
#         drop_old: bool = False,
#         **kwargs,
#     ):
#         self.name = "FacebookMyRocks"
#         self.db_config = db_config
#         self.case_config = db_case_config
#         self.db_name = "vectordbbench"
#         self.table_name = collection_name
#         self.dim = dim

#         # construct basic units
#         self.conn, self.cursor = self._create_connection(**self.db_config)

#         if drop_old:
#             self._drop_db()
#             self._create_db_table(dim)

#         self.cursor.close()
#         self.conn.close()
#         self.cursor = None
#         self.conn = None

#     @staticmethod
#     def _create_connection(**kwargs) -> tuple[mysql.connector.MySQLConnection, mysql.connector.cursor.MySQLCursor]:
#         conn = mysql.connector.connect(**kwargs)
#         cursor = conn.cursor()

#         assert conn is not None, "Connection is not initialized"
#         assert cursor is not None, "Cursor is not initialized"

#         return conn, cursor

#     def _drop_db(self):
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"
#         log.info(f"{self.name} client drop db : {self.db_name}")

#         # flush tables before dropping database to avoid some locking issue
#         self.cursor.execute("FLUSH TABLES")
#         self.cursor.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
#         self.cursor.execute("COMMIT")
#         self.cursor.execute("FLUSH TABLES")

#    

#     @contextmanager
#     def init(self):
#         """create and destoy connections to database.

#         Examples:
#             >>> with self.init():
#             >>>     self.insert_embeddings()
#         """
#         self.conn, self.cursor = self._create_connection(**self.db_config)

#         index_param = self.case_config.index_param()
#         search_param = self.case_config.search_param()

#         # Use database
#         self.cursor.execute(f"USE {self.db_name}")

#         # Set up SQL statements based on vector type and metric from config
#         vector_type = index_param["vector_type"]
#         metric_type = search_param["metric_type"]

#         if vector_type == "BLOB":
#             self.insert_sql = f"INSERT INTO {self.table_name} (id, v) VALUES (%s, FB_VECTOR_JSON_TO_BLOB(%s))"
#         else:  # JSON
#             self.insert_sql = f"INSERT INTO {self.table_name} (id, v) VALUES (%s, %s)"

#         # Build search SQL based on metric type
#         if metric_type == "L2":
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_L2(v, FB_VECTOR_JSON_TO_BLOB(%s))"
#             else:
#                 distance_expr = "FB_VECTOR_L2(v, %s)"
#             order_direction = "ASC"
#         elif metric_type == "IP":
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_IP(v, FB_VECTOR_JSON_TO_BLOB(%s))"
#             else:
#                 distance_expr = "FB_VECTOR_IP(v, %s)"
#             order_direction = "DESC"
#         elif metric_type == "COSINE":
#             # For cosine, use FB_VECTOR_NORMALIZE_L2 and FB_VECTOR_IP
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(FB_VECTOR_JSON_TO_BLOB(%s)))"
#             else:
#                 distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(%s))"
#             order_direction = "DESC"
#         else:
#             raise ValueError(f"Unsupported metric type: {metric_type}")

#         self.select_sql = (
#             f"SELECT id FROM {self.table_name} "
#             f"ORDER BY {distance_expr} {order_direction} "
#             f"LIMIT %s"
#         )

#         self.select_sql_with_filter = (
#             f"SELECT id FROM {self.table_name} WHERE id >= %s "
#             f"ORDER BY {distance_expr} {order_direction} "
#             f"LIMIT %s"
#         )

#         try:
#             yield
#         finally:
#             self.cursor.close()
#             self.conn.close()
#             self.cursor = None
#             self.conn = None

#     def ready_to_load(self) -> bool:
#         pass

#     def optimize(self) -> None:
#         """Optimization is handled during table creation for Facebook MyRocks"""
#         pass

#     @staticmethod
#     def vector_to_json(v) -> str:
#         """Convert vector to JSON string format"""
#         return json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)

#     # Removed manual normalization - using FB_VECTOR_NORMALIZE_L2 instead

#     def insert_embeddings(
#         self,
#         embeddings: list[list[float]],
#         metadata: list[int],
#         **kwargs,
#     ) -> tuple[int, Exception]:
#         """Insert embeddings into the database.
#         Should call self.init() first.
#         """
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         try:
#             batch_data = []
#             for i, embedding in enumerate(embeddings):
#                 vector_json = self.vector_to_json(embedding)
#                 batch_data.append((int(metadata[i]), vector_json))

#             self.cursor.executemany(self.insert_sql, batch_data)
#             self.cursor.execute("COMMIT")

#             return len(metadata), None
#         except Exception as e:
#             log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
#             return 0, e

#     def search_embedding(
#         self,
#         query: list[float],
#         k: int = 100,
#         filters: dict | None = None,
#         timeout: int | None = None,
#         **kwargs,
#     ) -> list[int]:
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         query_json = self.vector_to_json(query)

#         try:
#             if filters:
#                 self.cursor.execute(
#                     self.select_sql_with_filter,
#                     (filters.get("id"), query_json, k)
#                 )
#             else:
#                 self.cursor.execute(self.select_sql, (query_json, k))

#             return [id for (id,) in self.cursor.fetchall()]
#         except Exception as e:
#             log.warning(f"Failed to search embeddings: {e}")
#             return []
#------------------------------------------------------------------------------
# import logging
# from contextlib import contextmanager
# import json

# import mysql.connector
# import numpy as np

# from ..api import VectorDB
# from .config import FacebookMyRocksConfigDict, FacebookMyRocksIndexConfig

# log = logging.getLogger(__name__)


# class FacebookMyRocks(VectorDB):
#     def __init__(
#         self,
#         dim: int,
#         db_config: FacebookMyRocksConfigDict,
#         db_case_config: FacebookMyRocksIndexConfig,
#         collection_name: str = "vec_collection",
#         drop_old: bool = False,
#         **kwargs,
#     ):
#         self.name = "FacebookMyRocks"
#         self.db_config = db_config
#         self.case_config = db_case_config
#         self.db_name = "vectordbbench"
#         self.table_name = collection_name
#         self.dim = dim

#         # construct basic units
#         self.conn, self.cursor = self._create_connection(**self.db_config)

#         if drop_old:
#             self._drop_db()
#             self._create_db_table(dim)

#         self.cursor.close()
#         self.conn.close()
#         self.cursor = None
#         self.conn = None

#     @staticmethod
#     def _create_connection(**kwargs) -> tuple[mysql.connector.MySQLConnection, mysql.connector.cursor.MySQLCursor]:
#         conn = mysql.connector.connect(**kwargs)
#         cursor = conn.cursor()

#         assert conn is not None, "Connection is not initialized"
#         assert cursor is not None, "Cursor is not initialized"

#         return conn, cursor

#     def _drop_db(self):
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"
#         log.info(f"{self.name} client drop db : {self.db_name}")

#         # flush tables before dropping database to avoid some locking issue
#         self.cursor.execute("FLUSH TABLES")
#         self.cursor.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
#         self.cursor.execute("COMMIT")
#         self.cursor.execute("FLUSH TABLES")

#     def _create_db_table(self, dim: int):
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         index_param = self.case_config.index_param()

#         try:
#             log.info(f"{self.name} client create database : {self.db_name}")
#             self.cursor.execute(f"CREATE DATABASE {self.db_name}")

#             log.info(f"{self.name} client create table : {self.table_name}")
#             self.cursor.execute(f"USE {self.db_name}")

#             # Create auxiliary table for IVF indexes if needed
#             if index_param["index_type"] in ["ivfflat", "ivfpq"]:
#                 trained_index_table = index_param.get("trained_index_table")
#                 if trained_index_table:
#                     log.info(f"{self.name} creating auxiliary table for trained index: {trained_index_table}")
#                     self.cursor.execute(f"""
#                         CREATE TABLE IF NOT EXISTS {trained_index_table} (
#                             id VARCHAR(128) NOT NULL,
#                             type VARCHAR(128) NOT NULL,
#                             seqno INT NOT NULL,
#                             value JSON NOT NULL,
#                             PRIMARY KEY (id, type, seqno)
#                         ) ENGINE=rocksdb
#                     """)

#             # Get vector type from config
#             vector_type = index_param["vector_type"]
            
#             if vector_type == "BLOB":
#                 vector_column = f"v BLOB NOT NULL FB_VECTOR_DIMENSION {self.dim}"
#             else:  # JSON
#                 vector_column = f"v JSON NOT NULL FB_VECTOR_DIMENSION {self.dim}"

#             # Create table with vector index
#             index_type = index_param["index_type"]
#             create_table_sql = f"""
#                 CREATE TABLE {self.table_name} (
#                     id BIGINT NOT NULL PRIMARY KEY,
#                     {vector_column},
#                     INDEX vector_idx(v) FB_VECTOR_INDEX_TYPE '{index_type}'
#                 ) ENGINE=rocksdb
#             """

#             # Add additional index parameters for IVF types
#             if index_type in ["ivfflat", "ivfpq"]:
#                 trained_index_table = index_param.get("trained_index_table")
#                 trained_index_id = index_param.get("trained_index_id")
#                 if trained_index_table and trained_index_id:
#                     create_table_sql = create_table_sql.replace(
#                         f"FB_VECTOR_INDEX_TYPE '{index_type}'",
#                         f"FB_VECTOR_INDEX_TYPE '{index_type}' "
#                         f"FB_VECTOR_TRAINED_INDEX_TABLE '{trained_index_table}' "
#                         f"FB_VECTOR_TRAINED_INDEX_ID {trained_index_id}"
#                     )

#             self.cursor.execute(create_table_sql)
#             self.cursor.execute("COMMIT")

#         except Exception as e:
#             log.warning(f"Failed to create table: {self.table_name} error: {e}")
#             raise e from None

#     @contextmanager
#     def init(self):
#         """create and destoy connections to database.

#         Examples:
#             >>> with self.init():
#             >>>     self.insert_embeddings()
#         """
#         self.conn, self.cursor = self._create_connection(**self.db_config)

#         index_param = self.case_config.index_param()
#         search_param = self.case_config.search_param()

#         # Use database
#         self.cursor.execute(f"USE {self.db_name}")

#         # Set IVF search parameters if applicable
#         if hasattr(search_param, 'get') and search_param.get("nprobe"):
#             # Note: Facebook MyRocks may have session variables for IVF search
#             # This would need to be implemented based on actual FB MyRocks documentation
#             pass

#         # Set up SQL statements based on vector type and metric from config
#         vector_type = index_param["vector_type"]
#         metric_type = search_param["metric_type"]

#         if vector_type == "BLOB":
#             self.insert_sql = f"INSERT INTO {self.table_name} (id, v) VALUES (%s, FB_VECTOR_JSON_TO_BLOB(%s))"
#         else:  # JSON
#             self.insert_sql = f"INSERT INTO {self.table_name} (id, v) VALUES (%s, %s)"

#         # Build search SQL based on metric type
#         if metric_type == "L2":
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_L2(v, FB_VECTOR_JSON_TO_BLOB(%s))"
#             else:
#                 distance_expr = "FB_VECTOR_L2(v, %s)"
#             order_direction = "ASC"
#         elif metric_type == "IP":
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_IP(v, FB_VECTOR_JSON_TO_BLOB(%s))"
#             else:
#                 distance_expr = "FB_VECTOR_IP(v, %s)"
#             order_direction = "DESC"
#         elif metric_type == "COSINE":
#             # For cosine, use FB_VECTOR_NORMALIZE_L2 and FB_VECTOR_IP
#             if vector_type == "BLOB":
#                 distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(FB_VECTOR_JSON_TO_BLOB(%s)))"
#             else:
#                 distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(%s))"
#             order_direction = "DESC"
#         else:
#             raise ValueError(f"Unsupported metric type: {metric_type}")

#         self.select_sql = (
#             f"SELECT id FROM {self.table_name} "
#             f"ORDER BY {distance_expr} {order_direction} "
#             f"LIMIT %s"
#         )

#         self.select_sql_with_filter = (
#             f"SELECT id FROM {self.table_name} WHERE id >= %s "
#             f"ORDER BY {distance_expr} {order_direction} "
#             f"LIMIT %s"
#         )

#         try:
#             yield
#         finally:
#             self.cursor.close()
#             self.conn.close()
#             self.cursor = None
#             self.conn = None

#     def ready_to_load(self) -> bool:
#         """Check if the database is ready for data loading"""
#         return True

#     def needs_training(self) -> bool:
#         """Check if the index needs training"""
#         index_param = self.case_config.index_param()
#         return index_param["index_type"] in ["ivfflat", "ivfpq"]

#     def is_trained(self) -> bool:
#         """Check if IVF index is already trained"""
#         if not self.needs_training():
#             return True
            
#         index_param = self.case_config.index_param()
#         trained_index_table = index_param.get("trained_index_table")
#         trained_index_id = index_param.get("trained_index_id")
        
#         if not trained_index_table:
#             return False
            
#         try:
#             with self.init():
#                 self.cursor.execute(
#                     f"SELECT COUNT(*) FROM {trained_index_table} WHERE id = %s AND type = 'metadata'",
#                     (str(trained_index_id),)
#                 )
#                 result = self.cursor.fetchone()
#                 return result and result[0] > 0
#         except Exception:
#             return False

#     def optimize(self) -> None:
#         """Train IVF indexes and optimize the database"""
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         index_param = self.case_config.index_param()
        
#         # Only train for IVF indexes
#         if index_param["index_type"] in ["ivfflat", "ivfpq"]:
#             self._train_ivf_index()
        
#         log.info(f"{self.name} optimization completed")

#     def _train_ivf_index(self):
#         """Train IVF index using sample data and store in auxiliary table
        
#         IVF Training Process:
#         1. Sample training vectors (30-256 * nlist recommended)
#         2. Perform k-means clustering to generate centroids
#         3. For IVFPQ: Additional product quantization training
#         4. Store training results in auxiliary table
#         """
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         index_param = self.case_config.index_param()
#         trained_index_table = index_param.get("trained_index_table")
#         trained_index_id = index_param.get("trained_index_id")
        
#         if not trained_index_table:
#             log.warning("No trained index table specified for IVF training")
#             return

#         try:
#             log.info(f"{self.name} starting IVF index training")
            
#             # Sample training data (typically 30-256 * nlist vectors for good clustering)
#             nlist = index_param.get("nlist", 1000)
#             # Use recommended training size: between 30*nlist and 256*nlist
#             min_training_size = 30 * nlist
#             max_training_size = min(256 * nlist, 1000000)  # Cap at 1M vectors
#             training_sample_size = min(max_training_size, max(min_training_size, 10000))
            
#             log.info(f"Sampling {training_sample_size} vectors for IVF training (nlist={nlist})")
            
#             # Get total count to determine if we have enough data
#             self.cursor.execute(f"SELECT COUNT(*) FROM {self.db_name}.{self.table_name}")
#             total_count = self.cursor.fetchone()[0]
            
#             if total_count < nlist:
#                 log.warning(f"Dataset size ({total_count}) is smaller than nlist ({nlist}). Training may be suboptimal.")
            
#             # Sample vectors for training using reservoir sampling for better distribution
#             sample_sql = f"""
#                 SELECT v FROM {self.db_name}.{self.table_name} 
#                 ORDER BY RAND() 
#                 LIMIT {min(training_sample_size, total_count)}
#             """
            
#             self.cursor.execute(sample_sql)
#             sample_results = self.cursor.fetchall()
            
#             if not sample_results:
#                 log.warning("No data available for IVF training")
#                 return
                
#             log.info(f"Retrieved {len(sample_results)} sample vectors for training")
            
#             # Convert sample data for training
#             training_vectors = self._extract_training_vectors(sample_results, index_param["vector_type"])
            
#             if not training_vectors:
#                 log.warning("Failed to extract training vectors")
#                 return
                
#             log.info(f"Processed {len(training_vectors)} training vectors")
            
#             # Perform k-means clustering to generate centroids
#             actual_nlist = min(nlist, len(training_vectors))
#             centroids = self._perform_kmeans_clustering(training_vectors, actual_nlist)
            
#             if centroids is None:
#                 log.warning("K-means clustering failed")
#                 return
            
#             # For IVFPQ, generate additional product quantization data
#             pq_data = None
#             if index_param["index_type"] == "ivfpq":
#                 pq_data = self._train_product_quantization(training_vectors, index_param)
                
#             # Store training results in auxiliary table
#             self._store_training_results(trained_index_table, trained_index_id, centroids, index_param, pq_data)
            
#             log.info(f"{self.name} IVF index training completed successfully")
            
#         except Exception as e:
#             log.error(f"Failed to train IVF index: {e}")
#             raise e from None

#     def _extract_training_vectors(self, sample_results: list, vector_type: str) -> list[list[float]]:
#         """Extract training vectors from database results"""
#         training_vectors = []
        
#         try:
#             if vector_type == "BLOB":
#                 # Convert BLOB to vectors for training
#                 for (blob_data,) in sample_results:
#                     # Convert BLOB back to vector using FB_VECTOR_BLOB_TO_JSON
#                     self.cursor.execute("SELECT FB_VECTOR_BLOB_TO_JSON(%s)", (blob_data,))
#                     json_result = self.cursor.fetchone()
#                     if json_result and json_result[0]:
#                         import json
#                         vector = json.loads(json_result[0])
#                         if isinstance(vector, list) and len(vector) == self.dim:
#                             training_vectors.append(vector)
#             else:
#                 # JSON vectors can be used directly
#                 for (json_data,) in sample_results:
#                     import json
#                     if isinstance(json_data, str):
#                         vector = json.loads(json_data)
#                     else:
#                         vector = json_data
                    
#                     if isinstance(vector, list) and len(vector) == self.dim:
#                         training_vectors.append(vector)
                        
#         except Exception as e:
#             log.error(f"Failed to extract training vectors: {e}")
            
#         return training_vectors

#     def _train_product_quantization(self, training_vectors: list[list[float]], index_param: dict) -> dict | None:
#         """Train product quantization for IVFPQ index"""
#         try:
#             import numpy as np
            
#             m = index_param.get("m", 8)  # Number of subquantizers
#             nbits = index_param.get("nbits", 8)  # Bits per subquantizer
            
#             log.info(f"Training product quantization with m={m}, nbits={nbits}")
            
#             X = np.array(training_vectors, dtype=np.float32)
#             d = X.shape[1]  # Dimension
            
#             if d % m != 0:
#                 log.warning(f"Vector dimension ({d}) is not divisible by m ({m}). PQ training may be suboptimal.")
            
#             # Simple PQ training - in practice, this would use more sophisticated methods
#             subvector_dim = d // m
#             codebooks = []
            
#             for i in range(m):
#                 start_idx = i * subvector_dim
#                 end_idx = start_idx + subvector_dim
#                 subvectors = X[:, start_idx:end_idx]
                
#                 # K-means on subvectors
#                 from sklearn.cluster import KMeans
#                 kmeans = KMeans(n_clusters=2**nbits, random_state=42, n_init=5)
#                 kmeans.fit(subvectors)
                
#                 codebooks.append(kmeans.cluster_centers_.tolist())
            
#             pq_data = {
#                 "m": m,
#                 "nbits": nbits,
#                 "codebooks": codebooks,
#                 "subvector_dim": subvector_dim
#             }
            
#             log.info(f"Product quantization training completed")
#             return pq_data
            
#         except Exception as e:
#             log.error(f"Product quantization training failed: {e}")
#             return None

#     def _perform_kmeans_clustering(self, training_vectors: list[list[float]], nlist: int) -> list[list[float]] | None:
#         """Perform k-means clustering to generate centroids"""
#         try:
#             import numpy as np
#             from sklearn.cluster import KMeans
            
#             log.info(f"Starting k-means clustering with {nlist} clusters")
            
#             # Convert to numpy array
#             X = np.array(training_vectors, dtype=np.float32)
            
#             if len(X) < nlist:
#                 log.warning(f"Training data size ({len(X)}) is smaller than nlist ({nlist})")
#                 nlist = len(X)
            
#             # Perform k-means clustering
#             kmeans = KMeans(
#                 n_clusters=nlist,
#                 random_state=42,
#                 n_init=10,
#                 max_iter=300
#             )
            
#             kmeans.fit(X)
#             centroids = kmeans.cluster_centers_.tolist()
            
#             log.info(f"K-means clustering completed, generated {len(centroids)} centroids")
#             return centroids
            
#         except ImportError:
#             log.error("scikit-learn is required for k-means clustering. Please install it.")
#             return None
#         except Exception as e:
#             log.error(f"K-means clustering failed: {e}")
#             return None

#     def _store_training_results(self, trained_index_table: str, trained_index_id: int, 
#                               centroids: list[list[float]], index_param: dict, pq_data: dict | None = None):
#         """Store training results in the auxiliary table"""
#         try:
#             import json
            
#             log.info(f"Storing training results in {trained_index_table}")
            
#             # Clear any existing training data for this index
#             self.cursor.execute(
#                 f"DELETE FROM {trained_index_table} WHERE id = %s",
#                 (str(trained_index_id),)
#             )
            
#             # Store centroids
#             for i, centroid in enumerate(centroids):
#                 centroid_data = {
#                     "centroid_id": i,
#                     "centroid": centroid,
#                     "index_type": index_param["index_type"],
#                     "metric_type": index_param["metric_type"],
#                     "dimension": len(centroid)
#                 }
                
#                 self.cursor.execute(
#                     f"INSERT INTO {trained_index_table} (id, type, seqno, value) VALUES (%s, %s, %s, %s)",
#                     (str(trained_index_id), "centroid", i, json.dumps(centroid_data))
#                 )
            
#             # Store PQ data if available (for IVFPQ)
#             if pq_data:
#                 for i, codebook in enumerate(pq_data["codebooks"]):
#                     pq_codebook_data = {
#                         "subquantizer_id": i,
#                         "codebook": codebook,
#                         "subvector_dim": pq_data["subvector_dim"]
#                     }
                    
#                     self.cursor.execute(
#                         f"INSERT INTO {trained_index_table} (id, type, seqno, value) VALUES (%s, %s, %s, %s)",
#                         (str(trained_index_id), "pq_codebook", i, json.dumps(pq_codebook_data))
#                     )
            
#             # Store metadata
#             metadata = {
#                 "nlist": len(centroids),
#                 "index_type": index_param["index_type"],
#                 "metric_type": index_param["metric_type"],
#                 "vector_type": index_param["vector_type"],
#                 "training_completed": True,
#                 "training_vectors_count": len(centroids) * 50  # Estimate
#             }
            
#             if index_param["index_type"] == "ivfpq" and pq_data:
#                 metadata.update({
#                     "m": pq_data["m"],
#                     "nbits": pq_data["nbits"],
#                     "subvector_dim": pq_data["subvector_dim"]
#                 })
            
#             self.cursor.execute(
#                 f"INSERT INTO {trained_index_table} (id, type, seqno, value) VALUES (%s, %s, %s, %s)",
#                 (str(trained_index_id), "metadata", 0, json.dumps(metadata))
#             )
            
#             self.cursor.execute("COMMIT")
            
#             pq_info = f" and {len(pq_data['codebooks'])} PQ codebooks" if pq_data else ""
#             log.info(f"Successfully stored {len(centroids)} centroids{pq_info} and metadata")
            
#         except Exception as e:
#             log.error(f"Failed to store training results: {e}")
#             raise e from None

#     @staticmethod
#     def vector_to_json(v) -> str:
#         """Convert vector to JSON string format"""
#         return json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)

#     # Removed manual normalization - using FB_VECTOR_NORMALIZE_L2 instead

#     def insert_embeddings(
#         self,
#         embeddings: list[list[float]],
#         metadata: list[int],
#         **kwargs,
#     ) -> tuple[int, Exception]:
#         """Insert embeddings into the database.
#         Should call self.init() first.
#         """
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         try:
#             batch_data = []
#             for i, embedding in enumerate(embeddings):
#                 vector_json = self.vector_to_json(embedding)
#                 batch_data.append((int(metadata[i]), vector_json))

#             self.cursor.executemany(self.insert_sql, batch_data)
#             self.cursor.execute("COMMIT")

#             return len(metadata), None
#         except Exception as e:
#             log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
#             return 0, e

#     def search_embedding(
#         self,
#         query: list[float],
#         k: int = 100,
#         filters: dict | None = None,
#         timeout: int | None = None,
#         **kwargs,
#     ) -> list[int]:
#         assert self.conn is not None, "Connection is not initialized"
#         assert self.cursor is not None, "Cursor is not initialized"

#         query_json = self.vector_to_json(query)

#         try:
#             if filters:
#                 self.cursor.execute(
#                     self.select_sql_with_filter,
#                     (filters.get("id"), query_json, k)
#                 )
#             else:
#                 self.cursor.execute(self.select_sql, (query_json, k))

#             return [id for (id,) in self.cursor.fetchall()]
#         except Exception as e:
#             log.warning(f"Failed to search embeddings: {e}")
#             return []
#---------------------------------------------------------------------------------------------

import logging
import json

import numpy as np
from contextlib import contextmanager

from ..api import DBCaseConfig, EmptyDBCaseConfig, IndexType, VectorDB
from .config import FacebookMyRocksConfig

import mysql.connector
from mysql.connector import Error

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class FacebookMyRocks(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        print(db_config)

        self.user = db_config.get("user", "root")
        self.password = db_config.get("password")
        self.host = db_config.get("host", "localhost")
        self.database = db_config.get("database")
        self.vector_type = db_config.get("vector_type", "BLOB")
        self.metric_type = db_config.get("metric_type", "COSINE")
        self.index_type = db_config.get("index_type", "flat")
        self.table_name = "vec_collection"
        self.trained_index_table = "VECTORDB_DATA"
        self.trained_index_id = "cohere_wiki_ivfpq"
        self.dimension = db_config.get("dimension", 768)
        self.name = "FacebookMyRocks"

        # construct basic units
        self.conn, self.cursor = self._create_connection()


        self._drop_table(self.table_name)
        self._create_table(self.dimension)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None



    def _create_connection(self):
        try:
            # Create connection
            conn = mysql.connector.connect(
                host=self.host,       # MySQL server address
                user=self.user,            # MySQL username
                password=self.password, # MySQL password
                database=self.database      # Database name
            )

            if conn.is_connected():
                print("✅ Connected to MySQL database")

                # Create a cursor
                cursor = conn.cursor()

                assert cursor is not None, "Cursor is not initialized"

                return conn, cursor
            
        except Error as e:
            print("❌ Error while connecting to MySQL:", e)

    
    def _create_table(self, dim):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self.vector_type == "BLOB":
            vector_column = f"v BLOB NOT NULL FB_VECTOR_DIMENSION {dim}"
        else:
            vector_column = f"v JSON NOT NULL FB_VECTOR_DIMENSION {dim}"

        if self.index_type == "flat":
            create_table_sql = f"""
                CREATE TABLE {self.table_name} (
                    id BIGINT NOT NULL PRIMARY KEY,
                    {vector_column},
                    name VARCHAR(64) COLLATE utf8mb4_bin,
                    label VARCHAR(64) COLLATE utf8mb4_bin, -- added label field
                    INDEX vector_idx(v) FB_VECTOR_INDEX_TYPE '{self.index_type}'
                ) ENGINE=rocksdb
            """

        elif self.index_type in ["ivfflat", "ivfpq"]:
            create_table_sql = f"""
                CREATE TABLE {self.table_name} (
                    id BIGINT NOT NULL PRIMARY KEY,
                    {vector_column},
                    name VARCHAR(64) COLLATE utf8mb4_bin,
                    label VARCHAR(64) COLLATE utf8mb4_bin, -- added label field
                    INDEX vector_idx(v) FB_VECTOR_INDEX_TYPE '{self.index_type}' FB_VECTOR_TRAINED_INDEX_TABLE '{self.trained_index_table}' FB_VECTOR_TRAINED_INDEX_ID '{self.trained_index_id}'
                ) ENGINE=rocksdb
            """

        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def _drop_table(self, table_name):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table: {table_name}")

        # Flush tables before dropping to avoid locking issues
        self.cursor.execute("FLUSH TABLES")
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.conn.commit()  # Better to use connection's commit method
        self.cursor.execute("FLUSH TABLES")

    def _vector_to_json(self, v) -> str:
        """Convert vector to JSON string format"""
        return json.dumps(v.tolist() if isinstance(v, np.ndarray) else v)
    
    def _build_where_clause(self, filters: dict) -> tuple[str, list]:  # Remove the asterisks
        """Build WHERE clause and parameters from filters dict.
        
        Returns:
            tuple: (where_clause, params)
        """
        conditions = []
        params = []
        
        for field, condition in filters.items():
            if field == "id":
                if isinstance(condition, int):
                    conditions.append("id = %s")
                    params.append(condition)
                elif isinstance(condition, dict):
                    if "gte" in condition:
                        conditions.append("id >= %s")
                        params.append(condition["gte"])
                    if "lte" in condition:
                        conditions.append("id <= %s")
                        params.append(condition["lte"])
                    if "in" in condition:
                        placeholders = ",".join(["%s"] * len(condition["in"]))
                        conditions.append(f"id IN ({placeholders})")
                        params.extend(condition["in"])
            
            elif field == "label":  # This is correct if your schema has 'label'
                if isinstance(condition, str):
                    conditions.append("label = %s")
                    params.append(condition)
                elif isinstance(condition, dict):
                    if "like" in condition:
                        conditions.append("label LIKE %s")
                        params.append(condition["like"])
                    if "in" in condition:
                        placeholders = ",".join(["%s"] * len(condition["in"]))
                        conditions.append(f"label IN ({placeholders})")
                        params.extend(condition["in"])
        
        if not conditions:
            return "", []
        
        return "WHERE " + " AND ".join(conditions), params

    @classmethod
    def config_cls(cls) -> type[FacebookMyRocksConfig]:
        return FacebookMyRocksConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> type[DBCaseConfig]:
        return EmptyDBCaseConfig


    @contextmanager
    def init(self):
        """create and destoy connections to database."""
        self.conn, self.cursor = self._create_connection()

        if self.vector_type == "BLOB":
            self.insert_sql = f"INSERT INTO {self.table_name} (id, v, name, label) VALUES (%s, FB_VECTOR_JSON_TO_BLOB(%s), %s, %s)"
        else:  # JSON
            self.insert_sql = f"INSERT INTO {self.table_name} (id, v, name, label) VALUES (%s, %s, %s, %s)"

        # Build search SQL based on metric type
        if self.metric_type == "L2":
            if self.vector_type == "BLOB":
                self.distance_expr = "FB_VECTOR_L2(v, FB_VECTOR_JSON_TO_BLOB(%s))"
            else:
                self.distance_expr = "FB_VECTOR_L2(v, %s)"
            self.order_direction = "ASC"
        elif self.metric_type == "IP":
            if self.vector_type == "BLOB":
                self.distance_expr = "FB_VECTOR_IP(v, FB_VECTOR_JSON_TO_BLOB(%s))"
            else:
                self.distance_expr = "FB_VECTOR_IP(v, %s)"
            self.order_direction = "DESC"
        elif self.metric_type == "COSINE":
            # For cosine, use FB_VECTOR_NORMALIZE_L2 and FB_VECTOR_IP
            if self.vector_type == "BLOB":
                # self.distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(FB_VECTOR_JSON_TO_BLOB(%s)))"
                self.distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(%s))"
                print("Expr")
            else:
                self.distance_expr = "FB_VECTOR_IP(FB_VECTOR_NORMALIZE_L2(v), FB_VECTOR_NORMALIZE_L2(%s))"
            self.order_direction = "DESC"
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")

        # self.select_sql = (
        #     f"SELECT id FROM {self.table_name} "
        #     f"ORDER BY {distance_expr} {order_direction} "
        #     f"LIMIT %s"
        # )
        # Base SELECT template with placeholder for WHERE clause
        # Base SELECT template with placeholder for WHERE clause
        self.select_sql_template = (
            "SELECT id FROM {table_name} "
            "{WHERE_CLAUSE} "
            "ORDER BY {distance_expr} {order_direction} "
            "LIMIT %s"
        )

        # Pre-defined SQL for common cases (optimized)
        self.select_sql = self.select_sql_template.format(
            table_name=self.table_name,
            WHERE_CLAUSE="",
            distance_expr=self.distance_expr,  # Use the local variable
            order_direction=self.order_direction  # Use the local variable
        )

        self.select_sql_with_id_filter = self.select_sql_template.format(
            table_name=self.table_name,
            WHERE_CLAUSE="WHERE id >= %s",
            distance_expr=self.distance_expr,
            order_direction=self.order_direction
        )

        self.select_sql_with_label_filter = self.select_sql_template.format(
            table_name=self.table_name,
            WHERE_CLAUSE="WHERE label = %s",
            distance_expr=self.distance_expr,
            order_direction=self.order_direction
        )

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        assert len(embeddings) == len(metadata)

        try:
            batch_data = []
            for i in range(len(embeddings)):
                vector_json = self._vector_to_json(embeddings[i])
                name = f"cohere_vector_{metadata[i]}"
                label = labels_data[i] if labels_data else None
                batch_data.append((int(metadata[i]), vector_json, name, label))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            return 0, e
        

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        **kwargs,
    ) -> list[int]:
        
        if self.conn==None or self.cursor==None:
            self._create_connection()

        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        
        query_json = self._vector_to_json(query)
        
        try:
            # if filters:
            #     where_clause, filter_params = self._build_where_clause(filters)
            #     sql = self.select_sql_template.format(
            #         table_name=self.table_name,
            #         WHERE_CLAUSE=where_clause,
            #         distance_expr=self.distance_expr,  
            #         order_direction=self.order_direction
            #     )
            #     # Fix parameter order: filter params first, then query, then k
            #     params = [*filter_params, query_json, k]
            # else:
        #     sql = self.select_sql
        #     params = [query_json, k]
                
        #     self.cursor.execute(sql, params)
        #     return [id for (id,) in self.cursor.fetchall()]
            
        # except Exception as e:
        #     log.warning(f"Failed to search embeddings: {e}")
        #     return []
            sql = self.select_sql
            params = [query_json, k]
            
            log.debug(f"Executing SQL: {sql}")
            log.debug(f"With params: {params}")

            self.cursor.execute(sql, params)

            rows = self.cursor.fetchall()
            log.debug(f"Fetched rows: {rows}")

            ids = [id for (id,) in rows]
            log.debug(f"Extracted IDs: {ids}")

            return ids

        except Exception as e:
            log.warning(f"Failed to search embeddings: {e}", exc_info=True)
            return []




            