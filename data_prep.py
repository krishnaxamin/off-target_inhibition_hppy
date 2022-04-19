from pandas import DataFrame, concat, to_numeric, read_csv, merge, Series
from chembl_webresource_client.new_client import new_client

"""
Script to collect and clean data from ChEMBL and UKB. Data to be used for training and testing, and to make predictions
on. 
"""

# list of human and murine orthologs to happyhour (as found in FlyBase)
happyhour_orthologs = ['MAP4K3', 'MAP4K5', 'MAP4K2', 'MAP4K1']
ortholog_chemblids = []

# get list of ChEMBL IDs of the orthologs
target = new_client.target
for ortholog in happyhour_orthologs:
    query = DataFrame(target.search(ortholog))
    human_mouse_orthologs = query[(query['organism'] == 'Homo sapiens') | (query['organism'] == 'Mus musculus')]
    human_mouse_orthologs_ids = human_mouse_orthologs['target_chembl_id'].tolist()
    for id in human_mouse_orthologs_ids:
        ortholog_chemblids.append(id)

# get dataframe of all recorded bioactivity data against all four orthologs
activity = new_client.activity
activity_df_list = []
for ortho_id in ortholog_chemblids:
    bioactivity_df = DataFrame(activity.filter(target_chembl_id=ortho_id))
    activity_df_list.append(bioactivity_df)
happyhour_bioactivity = concat(activity_df_list)
happyhour_bioactivity.to_csv('data/happyhour_bioactivity_data.csv', index=False)

# filter dataframe by inhibition data, i.e. excluding interaction data
happyhour_inhibition_bioactivity = happyhour_bioactivity[(happyhour_bioactivity['standard_type'] == 'Activity') |
                                                         (happyhour_bioactivity['standard_type'] == 'Inhibition') |
                                                         (happyhour_bioactivity['standard_type'] == 'IC50') |
                                                         (happyhour_bioactivity['standard_type'] == '% residual kinase activity') |
                                                         (happyhour_bioactivity['standard_type'] == 'Residual activity')]
happyhour_inhibition_bioactivity.to_csv('data/happyhour_inhibitor_data.csv', index=False)

# make sure all standard_values are numeric
happyhour_inhibition_bioactivity['standard_value'] = to_numeric(happyhour_inhibition_bioactivity.standard_value)
happyhour_inhibition_bioactivity.to_csv('data/happyhour_inhibitor_data_numeric.csv', index=False)

# remove entries that have either no data (standard_value) or no structural information (canonical_smiles)
happyhour_inhibition_data_present = happyhour_inhibition_bioactivity[happyhour_inhibition_bioactivity.standard_value.notna()]
happyhour_inhibition_data_smiles_present = happyhour_inhibition_data_present[happyhour_inhibition_data_present.canonical_smiles.notna()].reset_index(drop=True)
happyhour_inhibition_data_smiles_present.to_csv('data/happyhour_inhibitor_data_smiles_present.csv', index=False)

# remove duplications
happyhour_inhibition_data_smiles_present_unique = \
    happyhour_inhibition_data_smiles_present.drop_duplicates(['canonical_smiles', 'target_chembl_id', 'standard_type',
                                                              'assay_chembl_id', 'standard_value', 'document_chembl_id',
                                                              'parent_molecule_chembl_id'])
happyhour_inhibition_data_smiles_present_unique.to_csv('data/happyhour_inhibition_data_cleaned.csv', index=False)

# classification
df = read_csv('data/happyhour_inhibition_data_cleaned.csv')
class_list = []
for i in range(len(df)):
    if df['standard_type'][i] == 'IC50':
        if df['standard_relation'][i] == '>':
            class_list.append(0)
        else:
            class_list.append(1)
    elif df['standard_type'][i] == 'Inhibition':
        if df['standard_value'][i] > 100:
            class_list.append(0)
        elif 70 <= df['standard_value'][i] <= 100 and df['standard_relation'][i] in ['=', '>', 'nan', '>=']:
            class_list.append(1)
        else:
            class_list.append(0)
    else:
        if df['standard_value'][i] <= 65:
            class_list.append(1)
        else:
            class_list.append(0)

df['classification'] = class_list

# get longest smiles
smiles = []

for i in df.canonical_smiles.tolist():
  cpd = str(i).split('.')
  cpd_longest = max(cpd, key = len)
  smiles.append(cpd_longest)

smiles = Series(smiles, name='canonical_smiles')
df['canonical_smiles'] = smiles

# drop duplicates wrt molecule identity, classification and target (copes with the same molecule having different
# activities against different orthologs
df = df.drop_duplicates(['molecule_chembl_id', 'canonical_smiles', 'classification', 'target_chembl_id'])
df.to_csv('data/happyhour_inhibitor_data_cleaned_classed.csv', index=False)

# get important info
df_slim = df[['molecule_chembl_id', 'canonical_smiles', 'classification']]
df_slim.to_csv('data/happyhour_inhibitor_data_cleaned_classed_slim.csv', index=False)

# make molecule.smi file for PaDEL
df_padel = df_slim[['canonical_smiles', 'molecule_chembl_id']]
df_padel.to_csv('padel/molecule.smi', sep='\t', index=False, header=False)

# read in fingerprints - create file of molecule_chembl_id, classification, fingerprints
fingerprints = read_csv('padel/descriptors_output.csv').sort_values(by='Name').reset_index(drop=True)
df_name_class = df[['molecule_chembl_id', 'classification']].sort_values(by='molecule_chembl_id').reset_index(drop=True)
df_name_class_fingerprints = concat([df_name_class, fingerprints], axis=1).drop(['Name'], axis=1)
df_name_class_fingerprints.to_csv('data/happyhour_inhibitor_name_class_fingerprints.csv', index=False)

# find molecules with different activities against different targets

# finds molecules that appear more than once -> 491 molecules do so
df_multiple_appearances = df[df.groupby('molecule_chembl_id').molecule_chembl_id.transform('count') > 1].sort_values(by='molecule_chembl_id')

df_multiple_appearances_slim = df_multiple_appearances[['molecule_chembl_id', 'target_pref_name', 'classification']]
# reduce to one entry multiple entries of molecules that have the same classification - we are interested in molecules
# that have different classifications within their repeated entries
df_multiple_appearances_slim_interest = df_multiple_appearances_slim.drop_duplicates(['molecule_chembl_id', 'classification'])
# molecules that have different classifications within their repeated entries will appear more than once now - these are
# selected -> 27 molecules have multiple entries showing different activities:
# 2 of these show different activity against the same target; 25 show different activities against different targets.
df_multiple_appearances_slim_interest = df_multiple_appearances_slim_interest[df_multiple_appearances_slim_interest.groupby('molecule_chembl_id').molecule_chembl_id.transform('count') > 1]
# get full info on these entries
df_interest = df[df.molecule_chembl_id.isin(df_multiple_appearances_slim_interest.molecule_chembl_id.values.tolist())].sort_values(by='molecule_chembl_id')
# get full info on the 2 that show different activity against the same target, and write to csv
df_interest1 = df_interest[df_interest.molecule_chembl_id.isin(['CHEMBL4564337', 'CHEMBL4640297'])]
df_interest1.to_csv('data/hppy_samemolecule_sametarget_diffclass.csv', index=False)
# get full info on the 25 that show different activity against different targets, and write to csv
df_interest2 = df_interest[(df_interest.molecule_chembl_id != 'CHEMBL4564337') & (df_interest.molecule_chembl_id != 'CHEMBL4640297')]
df_interest2.to_csv('data/hppy_samemolecule_difftarget_diffclass.csv', index=False)

######

# get all FDA-approved small molecule drugs
molecules = new_client.molecule
approved_drugs = DataFrame(molecules.filter(max_phase=4))
approved_drugs_small_mols = approved_drugs[approved_drugs['molecule_type'] == 'Small molecule']
approved_drugs_small_mols = approved_drugs_small_mols[approved_drugs_small_mols.molecule_structures.notna()].reset_index(drop=True)

smiles_list = []
for i in range(len(approved_drugs_small_mols)):
    smiles = approved_drugs_small_mols['molecule_structures'][i]['canonical_smiles']
    smiles_list.append(smiles)

approved_drugs_small_mols['canonical_smiles'] = smiles_list
approved_drugs_small_mols_slim = approved_drugs_small_mols[['canonical_smiles', 'molecule_chembl_id']]
approved_drugs_small_mols_slim.to_csv('padel/molecule.smi', sep='\t', index=False, header=False)

# PaDEL here, and rename file to active_small_molecule_drugs_name_fingerprints.csv

######

# list provided by Yizhou
ukb_drugs = read_csv('data/drug_ukb_dt_SMILES.csv')
ukb_drugs_notna = ukb_drugs[ukb_drugs.smiles.notna()].reset_index(drop=True)

ukb_smiles = []
for smile in ukb_drugs_notna.smiles.tolist():
    cpd = str(smile).split(', ')
    cpd_longest = max(cpd, key=len)
    cpd2 = str(cpd_longest).split('.')
    cpd_longest2 = max(cpd2, key=len)
    ukb_smiles.append(cpd_longest2)
ukb_drugs_notna['canonical_smiles'] = ukb_smiles

drug_names_cleaned = []
for drug_name in ukb_drugs_notna.Drug_curated.tolist():
    drug_names_cleaned.append(drug_name.replace(' ', '_'))
ukb_drugs_notna['drugs'] = drug_names_cleaned
ukb_drugs_notna.to_csv('data/drug_ukb_cleaned.csv', index=False)

ukb_drugs_smi = ukb_drugs_notna[['canonical_smiles', 'drugs']]
ukb_drugs_smi.to_csv('padel/molecule.smi', sep='\t', index=False, header=False)

# PaDEL here, and rename file to ukb_name_fingerprints.csv
