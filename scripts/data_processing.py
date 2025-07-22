"""
Enterprise-grade data processing pipeline for fraud detection with:
- Robust schema validation
- Memory optimization
- Parallel processing
- Data lineage tracking
- Atomic writes
"""

import pandas as pd
import numpy as np
from pandera import Check, DataFrameSchema
from pandera.typing import Series, DataFrame
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
from multiprocessing import cpu_count
import swifter  # For parallel processing
import hashlib
import tempfile
import json
from datetime import datetime
from config.constants import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SCHEMA_REGISTRY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define strict schemas
class InputSchema:
    transactions = DataFrameSchema({
        "user_id": Series[str],
        "signup_time": Series[datetime],
        "purchase_time": Series[datetime],
        "purchase_value": Series[float, Check.greater_than(0)],
        "device_id": Series[str],
        "source": Series[str],
        "browser": Series[str],
        "sex": Series[str, Check.isin(["M", "F"])],
        "age": Series[int, Check.in_range(13, 100)],
        "ip_address": Series[str],
        "class": Series[int, Check.isin([0, 1])]
    })

    ip_mapping = DataFrameSchema({
        "lower_bound_ip_address": Series[str],
        "upper_bound_ip_address": Series[str],
        "country": Series[str]
    })

class DataProcessor:
    def __init__(self, enable_validation: bool = True):
        self.enable_validation = enable_validation
        self.cpu_cores = max(1, cpu_count() - 1)

    def _validate_and_log(self, df: pd.DataFrame, schema: DataFrameSchema, dataset_name: str) -> pd.DataFrame:
        """Validate DataFrame against schema with detailed error reporting"""
        if not self.enable_validation:
            return df

        try:
            validated = schema.validate(df, lazy=True)
            logger.info(f"Schema validation passed for {dataset_name}")
            return validated
        except Exception as e:
            error_counts = {}
            for err in e.failure_cases:
                error_counts[err["column"]] = error_counts.get(err["column"], 0) + 1
            
            logger.error(f"Validation failed for {dataset_name}:")
            for col, count in error_counts.items():
                logger.error(f"  {col}: {count} errors")
            
            raise ValueError(f"Schema validation failed for {dataset_name}") from e

    def _load_with_progress(self, file_path: Path, chunksize: int = 10000) -> pd.DataFrame:
        """Memory-efficient loading with progress tracking"""
        try:
            chunks = []
            with pd.read_csv(file_path, chunksize=chunksize, low_memory=False) as reader:
                for i, chunk in enumerate(reader):
                    chunks.append(chunk)
                    if i % 10 == 0:
                        logger.info(f"Loaded {i * chunksize} rows from {file_path.name}")
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise

    def _clean_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction-specific cleaning operations"""
        # Convert timestamps with error handling
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        
        # Handle IP addresses
        df['ip_address'] = df['ip_address'].str.strip()
        
        # Remove invalid timestamps
        initial_count = len(df)
        df = df[df['purchase_time'].notna() & df['signup_time'].notna()]
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with invalid timestamps")
        
        return df

    def _process_chunk(self, chunk: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Process a data chunk with validation and cleaning"""
        try:
            # Dataset-specific processing
            if dataset_name == "transactions":
                chunk = self._clean_transactions(chunk)
                schema = InputSchema.transactions
            elif dataset_name == "ip_mapping":
                schema = InputSchema.ip_mapping
            else:  # creditcard
                schema = SCHEMA_REGISTRY["creditcard"]
            
            return self._validate_and_log(chunk, schema, dataset_name)
        except Exception as e:
            logger.error(f"Failed processing chunk for {dataset_name}: {str(e)}")
            raise

    def _atomic_write(self, df: pd.DataFrame, output_path: Path):
        """Write data with atomic operation guarantee"""
        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_path = Path(tmp.name)
                df.to_parquet(temp_path, engine='pyarrow')
            
            # Atomic rename
            temp_path.replace(output_path)
            logger.info(f"Successfully wrote {len(df)} rows to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write data: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def process_dataset(self, file_path: Path, dataset_name: str) -> Path:
        """End-to-end processing for a single dataset"""
        try:
            logger.info(f"Starting processing for {dataset_name}")
            
            # Load data
            raw_df = self._load_with_progress(file_path)
            
            # Parallel processing
            chunks = np.array_split(raw_df, self.cpu_cores)
            processed_chunks = []
            
            with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
                futures = [
                    executor.submit(
                        self._process_chunk,
                        chunk,
                        dataset_name
                    ) for chunk in chunks
                ]
                
                for future in as_completed(futures):
                    processed_chunks.append(future.result())
            
            # Combine and deduplicate
            final_df = pd.concat(processed_chunks, ignore_index=True)
            final_df = final_df.drop_duplicates()
            
            # Generate output path
            output_path = PROCESSED_DATA_PATH / f"{dataset_name}_{datetime.now().strftime('%Y%m%d')}.parquet"
            
            # Atomic write
            self._atomic_write(final_df, output_path)
            
            # Data lineage
            self._record_lineage(file_path, output_path, final_df)
            
            return output_path
        except Exception as e:
            logger.error(f"Critical failure processing {dataset_name}: {str(e)}")
            raise

    def _record_lineage(self, input_path: Path, output_path: Path, df: pd.DataFrame):
        """Record data provenance information"""
        lineage = {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path.absolute()),
            "output_file": str(output_path.absolute()),
            "row_count": len(df),
            "columns": list(df.columns),
            "data_hash": hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        }
        
        lineage_path = output_path.with_suffix('.lineage.json')
        with open(lineage_path, 'w') as f:
            json.dump(lineage, f, indent=2)

    def run_pipeline(self) -> Dict[str, Path]:
        """Execute full processing pipeline"""
        datasets = {
            "transactions": RAW_DATA_PATH / "Fraud_Data.csv",
            "ip_mapping": RAW_DATA_PATH / "IpAddress_to_Country.csv",
            "creditcard": RAW_DATA_PATH / "creditcard.csv"
        }
        
        results = {}
        for name, path in datasets.items():
            try:
                results[name] = self.process_dataset(path, name)
            except Exception as e:
                logger.error(f"Pipeline aborted due to {name} processing failure")
                raise RuntimeError(f"Failed to process {name}") from e
        
        return results