import pandas as pd

POLYMER_ENTITIES_COLS = ('pe_index', 'sequence', 'poly_type') #columns for poly entities (and not common among other chains of same entry)
VERBOSE = True

def preprocess_pdb_data(input_path, output_path, filter_non_proteins):
    
    df = pd.read_csv(input_path)
    
    #df_pe is the poly-entities df with a row for each chain 
    df_pe = df.copy()
    
    #we will get a df of entries alone by leaving only entry-derived columns and dropping duplicates
    cols = list(df.columns)
    cols = [col for col in cols if col not in POL]
    df_entries = df[cols].drop_duplicates() 
    
    #filter non X-ray data
    
    n_pe = len(df_pe)
    n_entries = len(df_entries) 
    
    df_pe = df_pe.query('struct_method == "X-RAY DIFFRACTION"')
    df_entries = df_entries.query('struct_method == "X-RAY DIFFRACTION"')
    
    if VERBOSE:
        print(f'{n_pe - len(df_pe)} poly-entities with struct_method != ""X-RAY DIFFRACTION"" were removed!')
        print(f'{n_entries - len(df_entries)} entries with struct_method != ""X-RAY DIFFRACTION"" were removed!')
        
    
    if filter_non_proteins:
        
        mixed_ids = set(df_pe.query('poly_type != "Protein"').pdb_id)
        proteins_only_ids = set([id for id in df_pe.pdb_id if id not in mixed_ids])
        
        n_pe = len(df_pe)
        n_entries = len(df_entries)
        
        df_pe = df_pe[df_pe.pdb_id.isin(proteins_only_ids)]
        df_entries = df_entries[df_entries.pdb_id.isin(proteins_only_ids)]
        
        if VERBOSE:
            print(f'{n_pe - len(df_pe)} poly-entities in entries with non-protein entities were removed!')
            print(f'{n_entries - len(df_entries)} entries with non-protein entities were removed!')
            
    
    
     
    
    
    