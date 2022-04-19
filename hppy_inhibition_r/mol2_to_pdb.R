library(bio3d)

"
Script to convert mol2 file format into pdb file format.
"

# predicted inhibitors
losartan_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/losartan.mol2.gz')
losartan_pdb <- as.pdb(losartan_mol2)
write.pdb(losartan_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/losartan.pdb')

tegretol_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/tegretol.mol2.gz')
tegretol_pdb <- as.pdb(tegretol_mol2)
write.pdb(tegretol_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/tegretol.pdb')

# known inhibitor
staurosporine_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/staurosporine.mol2.gz')
staurosporine_pdb <- as.pdb(staurosporine_mol2)
write.pdb(staurosporine_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/staurosporine.pdb')

# looking at the pdb file for 5J5T
receptor_pdb <- read.pdb('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/5j5t_connected.pdb')

# known non-inhibitor
sr3677_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/sr-3677.mol2.gz')
sr3677_pdb <- as.pdb(sr3677_mol2)
write.pdb(sr3677_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/sr-3677.pdb')

# predicted non-inhibitor
diclofenac_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/diclofenac.mol2.gz')
diclofenac_pdb <- as.pdb(diclofenac_mol2)
write.pdb(diclofenac_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/diclofenac.pdb')

# known non-interactor
bms_mol2 <- read.mol2('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/bms.mol2.gz')
bms_pdb <- as.pdb(bms_mol2)
write.pdb(bms_pdb, file = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/bms.pdb')

# general 
path = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/ligands/'
mol2_path = paste(path)
mol2_files <- list.files(path = 'C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/ligands/mol2/', full.names = TRUE)
for (file in mol2_files) {
  mol <- strsplit(tail(strsplit(file, '/')[[1]], 1), '.mol2')[[1]][1]
  print(mol)
  mol2 <- read.mol2(file)
  pdb <- as.pdb(mol2)
  write.pdb(pdb, file = paste('C:/Users/krish/Documents/Cambridge/2021-2022/SysBio/Project/structures/ligands/', mol, '.pdb', 
                              sep = ''))
}
