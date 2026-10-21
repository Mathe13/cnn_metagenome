import torch
from torch.utils.data import IterableDataset, get_worker_info
import cudf
import pyarrow.parquet as pq # Essencial para ler metadados sem carregar o arquivo
import random 
import math
import warnings
import pandas as pd  # <--- Esta é a linha que faltou
import pyarrow.parquet as pq
import torch
import math
import random
import numpy as np
import warnings
from torch.utils.data import IterableDataset, get_worker_info
# Filtra avisos do PyTorch sobre IterableDataset length
warnings.filterwarnings("ignore", message="Length of IterableDataset.*")

class DaskCuDFIntegerDataset(IterableDataset):
    """
    Dataset iterável otimizado para PyTorch com suporte a Multi-Worker Loading.
    """
    def __init__(self, file_paths: list[str], padded_column_name: str, label_column: str,
                 maxlen: int, nucleotides: list, batch_size: int, debug: bool = False):
        self.file_paths = file_paths
        self.padded_column_name = padded_column_name
        self.label_column = label_column
        self.maxlen = maxlen
        self.nucleotides = nucleotides
        self.batch_size = batch_size
        self.debug = debug
        
        # Mapeamento: 'A' -> 0, 'C' -> 1, etc. 'N' ou Pad -> num_features
        self.mapping = {n: i for i, n in enumerate(self.nucleotides)}
        self.num_features = len(self.nucleotides)
        
        # Otimização 1: Calcular total de amostras sem carregar dados na GPU
        self.total_samples = self._calculate_total_samples_fast()
    
    def _calculate_total_samples_fast(self):
        """Lê apenas os metadados dos arquivos Parquet (CPU-bound, Baixa latência)."""
        count = 0
        try:
            for f in self.file_paths:
                # PyArrow lê apenas o footer do arquivo, instantâneo
                count += pq.read_metadata(f).num_rows
            return count
        except Exception as e:
            print(f"Aviso: Falha ao ler metadados com PyArrow: {e}")
            return None

    def __len__(self):
        """Retorna o número estimado de LOTES (Batches)."""
        if self.total_samples is None:
            # Fallback seguro
            return 1000 
        return math.ceil(self.total_samples / self.batch_size)
        
    def __iter__(self):
        # Otimização 2: Lógica de Multi-Processing (Worker Splitting)
        # Se isso não existir, cada worker lê TODOS os arquivos (duplicando dados)
        worker_info = get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            my_files = self.file_paths
        else:
            # Multi-process data loading: dividir arquivos entre workers
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            
            my_files = self.file_paths[iter_start:iter_end]
            
            if self.debug:
                print(f"Worker {worker_id} processando {len(my_files)} arquivos.")

        # Shuffle local (dentro da fatia do worker)
        # Importante para não ler sempre na mesma ordem em cada época
        random.shuffle(my_files)
        
        for file_path in my_files:
            try:
                # Leitura GPU (cuDF)
                partition_df = cudf.read_parquet(file_path)
            except Exception as e:
                print(f"ERRO CRÍTICO: Falha ao ler {file_path}: {e}")
                continue
            
            num_rows = len(partition_df)
            
            # Itera sobre o DataFrame em batches
            for i in range(0, num_rows, self.batch_size):
                end_index = min(i + self.batch_size, num_rows)
                
                # Slicing no cuDF é zero-copy (barato)
                batch_df = partition_df.iloc[i:end_index]
                
                try:
                    inputs_series = batch_df[self.padded_column_name]
                    labels_series = batch_df[self.label_column]


                    inputs_list = inputs_series.to_arrow().to_pylist()

                    tokenized_list = [
                        [self.mapping.get(n, self.num_features) for n in s] 
                        for s in inputs_list
                    ]
                    inputs_tensor = torch.tensor(tokenized_list, dtype=torch.long) # Default é CPU

                    labels_cpu = labels_series.to_numpy() 
                    labels_tensor = torch.as_tensor(labels_cpu, dtype=torch.long)
                    
                    yield inputs_tensor, labels_tensor

                except Exception as e:
                    print(f"ERRO no processamento do lote {i} em {file_path}: {e}")
                    continue

# Filtra avisos do PyTorch
warnings.filterwarnings("ignore", message="Length of IterableDataset.*")

class ParquetCPUDataset(IterableDataset):
    """
    Versão corrigida: Lê dados na CPU (Pandas/PyArrow) para evitar
    criação de contextos CUDA nos workers.
    """
    def __init__(self, file_paths: list[str], padded_column_name: str, label_column: str,
                 maxlen: int, nucleotides: list, batch_size: int, debug: bool = False):
        self.file_paths = file_paths
        self.padded_column_name = padded_column_name
        self.label_column = label_column
        self.maxlen = maxlen
        self.nucleotides = nucleotides
        self.batch_size = batch_size
        self.debug = debug
        
        # Mapeamento
        self.mapping = {n: i for i, n in enumerate(self.nucleotides)}
        self.num_features = len(self.nucleotides)
        
        # Otimização: Calcular total de amostras (CPU-bound)
        self.total_samples = self._calculate_total_samples_fast()
    
    def _calculate_total_samples_fast(self):
        count = 0
        try:
            for f in self.file_paths:
                count += pq.read_metadata(f).num_rows
            return count
        except Exception as e:
            print(f"Aviso: Falha ao ler metadados: {e}")
            return None

    def __len__(self):
        if self.total_samples is None:
            return 1000 
        return math.ceil(self.total_samples / self.batch_size)
        
    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:
            my_files = self.file_paths
        else:
            # Divide arquivos entre workers
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths))
            my_files = self.file_paths[iter_start:iter_end]
        
        # Shuffle local é importante
        random.shuffle(my_files)
        
        for file_path in my_files:
            try:
                # --- A GRANDE MUDANÇA: USAR PANDAS (CPU) ---
                # Usamos engine='pyarrow' que é muito rápido
                partition_df = pd.read_parquet(file_path, engine='pyarrow', columns=[self.padded_column_name, self.label_column])
            except Exception as e:
                print(f"ERRO CRÍTICO: Falha ao ler {file_path}: {e}")
                continue
            
            num_rows = len(partition_df)
            
            # Itera em batches
            for i in range(0, num_rows, self.batch_size):
                end_index = min(i + self.batch_size, num_rows)
                
                # Pandas slice (CPU)
                batch_df = partition_df.iloc[i:end_index]
                
                try:
                    inputs_list = batch_df[self.padded_column_name].tolist()
                    labels_list = batch_df[self.label_column].to_numpy() # ou .tolist()

                    # Tokenização na CPU (Python puro é rápido o suficiente para DNA)
                    tokenized_list = [
                        [self.mapping.get(n, self.num_features) for n in s] 
                        for s in inputs_list
                    ]
                    
                    # Cria Tensores na CPU (Pinned Memory fará o upload depois)
                    inputs_tensor = torch.tensor(tokenized_list, dtype=torch.long)
                    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
                    
                    yield inputs_tensor, labels_tensor

                except Exception as e:
                    print(f"ERRO no processamento do lote {i} em {file_path}: {e}")
                    continue