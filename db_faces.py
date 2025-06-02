from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. Connect ke Milvus
connections.connect("default", host="localhost", port="19530")

# 2. Definisikan schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="nim", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="nama", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="encoding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="type_image", dtype=DataType.VARCHAR, max_length=20)
]

schema = CollectionSchema(fields, description="Face vector database mahasiswa dengan tipe original/augmented")

# 3. Buat koleksi
collection_name = "faces"
collection = Collection(name=collection_name, schema=schema)

# 4. Tampilkan info sukses
print(f"✅ Koleksi '{collection_name}' berhasil dibuat dengan schema:")
for field in fields:
    print(f"- {field.name} ({field.dtype})")

# 4. Buat index untuk field ‘encoding’
index_params = {
    "index_type": "HNSW",   # bisa juga HNSW, etc.
    "metric_type": "COSINE",        # atau "IP" (inner product)
    "params": {"nlist": 128}    # tuning parameter untuk IVF
}
collection.create_index(field_name="encoding", index_params=index_params)
print(f"✅ Index pada field 'encoding' berhasil dibuat.")

# 5. (Opsional) Load koleksi untuk siap search
collection.load()
print(f"✅ Koleksi '{collection_name}' siap untuk operasi search.")
