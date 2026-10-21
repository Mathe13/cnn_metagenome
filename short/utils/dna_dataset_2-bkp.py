import torch
from torch.utils.data import IterableDataset
import cudf
import cupy as cp
import numpy as np
import os
import random 
import warnings
warnings.filterwarnings("ignore", message="Length of IterableDataset.*")


class DaskCuDFIntegerDataset(IterableDataset):
    """
    Um Dataset iterável otimizado para PyTorch que lê arquivos Parquet
    (partições Dask/cuDF) e retorna tensores de inteiros (IDs de nucleotídeos)
    e seus rótulos, em lotes.
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
        # Mapeamento de caractere para ID inteiro
        self.mapping = {n: i for i, n in enumerate(self.nucleotides)}
        self.num_features = len(self.nucleotides)
        
        # Novo atributo para armazenar o número total de amostras
        self.total_samples = 0
        self._calculate_total_samples()
   

    def _calculate_total_samples_old(self):
        """Calcula o número total de amostras lendo os metadados dos arquivos Parquet."""
        try:
            # Use pandas para ler os metadados sem carregar todos os dados na memória
            import pandas as pd
            for file_path in self.file_paths:
                self.total_samples += pd.read_parquet(file_path).shape[0]
        except ImportError:
            print("Aviso: Pandas não está instalado. Não foi possível calcular o tamanho do dataset.")
            self.total_samples = None
        except Exception as e:
            print(f"Aviso: Falha ao calcular o tamanho do dataset. Erro: {e}")
            self.total_samples = None


    def _calculate_total_samples(self):
        try:
            for file_path in self.file_paths:
                self.total_samples += cudf.read_parquet(file_path).shape[0]
        except Exception as e:
            print(f"Aviso: Falha ao calcular o tamanho do dataset. Erro: {e}")
            self.total_samples = None

    
    def __len__(self):
        """Retorna o número de lotes no dataset."""
        if self.total_samples is None:
            # Se não foi possível calcular o total, retorne um valor grande para o tqdm
            return 17062701// self.batch_size # Uma estimativa razoável
            
        from math import ceil
        return ceil(self.total_samples / self.batch_size)

        
    def __iter__(self):
        worker_file_paths = self.file_paths
        random.shuffle(worker_file_paths)
        
        for file_path in worker_file_paths:
            try:
                partition_df = cudf.read_parquet(file_path)
            except Exception as e:
                print(f"ERRO: Falha ao ler o arquivo {file_path}. Erro: {e}")
                continue
            
            for i in range(0, len(partition_df), self.batch_size):
                end_index = min(i + self.batch_size, len(partition_df))
                batch_df = partition_df.iloc[i:end_index]
                
                inputs_series = batch_df[self.padded_column_name]
                labels_series = batch_df[self.label_column]
                
                try:
                    # --- MUDANÇA NA LÓGICA DE TOKENIZAÇÃO ---
                    inputs_list = inputs_series.to_arrow().to_pylist()
                    tokenized_list = [[self.mapping.get(n, self.num_features) for n in s] for s in inputs_list]
                    inputs_tensor = torch.as_tensor(tokenized_list, device='cuda').to(torch.long).reshape(-1, self.maxlen)
                    # --- FIM DA MUDANANÇA ---

                    labels_tensor = torch.as_tensor(labels_series.values, device='cuda').to(torch.long)
                    
                    yield inputs_tensor, labels_tensor
                except Exception as e:
                    print(f"ERRO: Falha na tokenização/mapeamento no arquivo {file_path}, lote {i}. Erro: {e}")
                    continue