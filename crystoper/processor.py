import numpy as np
import pandas as pd
import torch
import re
from crystoper.utils.general import vprint
from crystoper import config

POLYMER_ENTITIES_COLS = ('pe_index', 'sequence', 'poly_type') #columns for poly entities (and not common among other chains of same entry)
FINAL_COLUMNS_ORDER = ['pdb_id', 'pe_index', 'sequence', 'poly_type', 'struct_method',	'crystal_method', 'ph',
                       'temp', 'pdbx_details', 'deposit_date',	'revision_date']

VERBOSE = True

def get_entries_df(df):
    "Get entries from the full data poly entities data"
    
    cols = list(df.columns)
    cols = [col for col in cols if col not in POLYMER_ENTITIES_COLS]
    df_entries = df[cols].drop_duplicates() 
    
    return df_entries

def preprocess_pdb_data(input_path, output_path,
                        **kwargs):

    df = pd.read_csv(input_path)
    
    df = filter_pdb_data(df, **kwargs)
    
    df = parse_ph_and_temperature(df, **kwargs)
    
    df = standardize_crystal_method(df)
    
    print_missing_report(df)
    
    #reorder    
    cols = [col for col in FINAL_COLUMNS_ORDER if col in df.columns]
    df = df[cols]
    
    return df
    
def filter_pdb_data(df,
                    filter_non_proteins,
                    chains_per_entry,
                    filter_empty_details,
                    minimum_details_length,
                    maximum_details_length,
                    **kwargs):

    #save original lengths for later printings
    original_n_entries = len(set(df.pdb_id))
    original_n_poly_entities = len(df)
    
    #df_pe is the poly-entities df with a row for each chain 
    df_pe = df[['pdb_id'] + list(POLYMER_ENTITIES_COLS)]
    
    #we will get a df of entries alone by leaving only entry-derived columns and dropping duplicates
    df_entries = get_entries_df(df)
    
    #filter non X-ray data
    
    n_entries = len(df_entries) 
    df_entries = df_entries.query('struct_method == "X-RAY DIFFRACTION"')
    
    vprint(f'{n_entries - len(df_entries)} entries with struct_method != ""X-RAY DIFFRACTION"" were removed!')
        
    
    if filter_non_proteins:
        
        mixed_ids = set(df_pe.query('poly_type != "Protein"').pdb_id)
        proteins_only_ids = set([id for id in df_pe.pdb_id if id not in mixed_ids])
        
        n_entries = len(df_entries)
        
        df_entries = df_entries[df_entries.pdb_id.isin(proteins_only_ids)]
        
        vprint(f'{n_entries - len(df_entries)} entries with non-protein entities were removed!')
    
    if chains_per_entry != [0]:
        
        s = df_pe.groupby('pdb_id').size()
        filtered_ids = set(s[s.isin(chains_per_entry)].index)
        
        df_entries = df_entries[df_entries.pdb_id.isin(filtered_ids)]
        
        vprint(f'{len(filtered_ids)} entries were filtered out due to number of chains (polymer entity) different from user input')
    
    
    if filter_empty_details:
        
        n_entries = len(df_entries)
        df_entries = df_entries.query('pdbx_details.notna()')
        
        vprint(f'{n_entries - len(df_entries)} entries with no "pdbx_details" were removed!')
        
                
    #re-merge data (it will be filtered according to the entries left in df_entries due to left merge)
    df = df_entries.merge(df_pe, how='left')
    
    df = filter_by_pdbx_details_length(df, minimum_details_length, maximum_details_length)
    
    vprint("\n\nTotal filtered values:")
    
    entries_filtered = original_n_entries - len(df_entries)
    pe_filtered = original_n_poly_entities - len(df)
    
    vprint(f'{entries_filtered} entries were filtered out of {original_n_entries} ({100*entries_filtered/len(df_entries):.2f}%) ')
    vprint(f'{pe_filtered} poly entities were filtered out of {original_n_poly_entities} ({100*pe_filtered/len(df):.2f}%) ')
    
    vprint('\n')
    vprint('Final data size:')
    vprint(f"Entries: {len(df_entries)}")
    vprint(f"Poly entities: {len(df)}")
        
    return df

def parse_ph_and_temperature(df, parse_ph, parse_temperature, **kwargs):
    
    if parse_ph:
        
        m = df.ph.isna()
        parsed_ph = df.loc[m, 'pdbx_details'].apply(parse_ph_from_string)
        df.loc[m, 'ph'] = parsed_ph

        n = parsed_ph.notna().sum()
         
        vprint(f'{parsed_ph.notna().sum()} missing pH values were parsed from the "pdbx_details" feature')
        
    if parse_temperature:
        
        m = df.temp.isna()
        parsed_temp = df.loc[m, 'pdbx_details'].apply(parse_temp_from_string)
        df.loc[m, 'temp'] = parsed_temp

        vprint(f'{parsed_temp.notna().sum()} missing temperature values were parsed from the "pdbx_details" feature')
            
    return df
                



def print_missing_report(df):
    
    vprint('')
    vprint('*'*50)
    vprint("** Missing values report for the full data (poly entities data): **")
    
    n = len(df)

    df = df.isna().sum().to_frame()
    df.columns = ['missing']
    df['total'] = n
    df['pct_missing'] = (100 * df.missing / df.total).round(2).astype(str) + '%'
    
    vprint(df.to_string())    

def parse_ph_from_string(s):
    s = s.lower()
    matches =  re.findall(r'ph\s*=*\s*?\d+(?:\.\d+)?', s)
    
    if len(matches) >=1 :
        v = matches[0]
        v = v.replace('ph', '').replace('=', '').replace(":", "").strip()
        v = float(v)
        if not 0.1 < v < 14:
            v = np.nan
            
        return v
        
    else:
        return np.nan
    
def parse_temp_from_string(s):
    s = s.lower()
    matches =  re.findall(r'temperature \d\d\dK?', s)
    
    if len(matches) > 0:
        return int(matches[0].split()[-1])
    else:
        return np.nan
    
def process_crystal_method(v):
    
    if not v:
        v = ''
    
    if v == '':
       return 'SITTING-DROP'
    
    v = v.lower()
    
    if 'evaporation' in v:
        return 'EVAPORATION'
    
    if 'cell' in v:
        return 'IN-CELL'
    
    if 'tubes' in v:
        return 'SMALL-TUBES'
    
    if 'batch' in v or 'bath' in v or 'microbach' in v:
        return 'BATCH'
    
    if 'seed' in v:
        return 'SEEDING'
    
    if 'cubic' in v or 'lcp' in v:
        return 'LCP'
    
    if 'dial' in v:
        return 'DIALYSIS'
    
    if 'hang' in v:
        return 'HANGING-DROP'
    
    if 'sit' in v:
        return 'SITTING-DROP'
    
    if 'liquid' in v:
        return 'LIQUID-DIFFUSION'
    
    if 'counter' in v:
        return 'COUNTER-DIFFUSION'
    
    if 'oil' in v:
        return 'UNDER-OIL'
    
    if 'slow cool' in v:
        return 'SLOW-COOL'
    
    #if vapor exists but not 'sitting' we assume sitting drop
    if 'vapor' in v:
        return 'SITTING-DROP'
    
    else:
        print(v)
        return np.nan
    
def standardize_crystal_method(df):
    
    df['crystal_method'] = df['crystal_method'].fillna('').apply(process_crystal_method)
    
    return df

def filter_for_single_entities(df):
    count = df.groupby('pdb_id').size().reset_index()
    singles = set(count[count[0]==1].pdb_id.values)
    df = df[df.pdb_id.isin(singles)]
    
    return df

def filter_by_pdbx_details_length(df, min_len, max_len):
    df = df[df.pdbx_details.str.len().between(min_len,max_len)]
    return df
