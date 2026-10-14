# GenomeOcean fine-tuning pipeline

This project first fixes the entire labeled dataset into `train 80% / validation 10% / inference 10%`, then fine-tunes separate GenomeOcean main and sub models and evaluates them on the final 10%.

- main: `0=Cellular`, `1=Viral`
  - Cellular: ARC, BAC, EUK, MITO, PLASTID
  - Viral: NCLDV, EVE NCLDV, Mirus, EVE Mirus, PHAGE
- sub: `0=NCLDV`, `1=Mirus`, `2=PHAGE`
  - NCLDV: conventional NCLDV + EVE NCLDV
  - Mirus: conventional Mirus + EVE Mirus
  - PHAGE: PHAGE snapshot
- taxonomy: an optional secondary task performed after primary classification

- Original FASTA files are used in read-only mode. This pipeline does not read the MetaVR DuckDB.
- No symbolic links are created, and all generated files are stored within this project.
- Internal project paths are resolved relative to the script location, so the entire folder can be moved elsewhere and still work.
- Only the original data paths in `configs/data_sources.yaml` are absolute. `data/preprocessed` is the final preprocessing output location specified by the user.
- NCLDV/Mirus/EVE records with insufficient taxonomy information can still be used for main/sub classification. Records without taxonomy target labels are excluded from the auxiliary taxonomy training.

## 1. Directory structure and responsibilities

```text
02_gv_genomeocean_holdout/
├── README.md                 # Complete documentation in this single file
├── requirement.txt          # Python package versions
├── configs/
│   ├── data_sources.yaml    # Original absolute paths and label/group rules for each source
│   └── pipeline.yaml        # All configurable preprocessing, split, training, and evaluation hyperparameters
├── scripts/
│   ├── 01_preprocessing/    # Manifest → similarity checks → split → 5 kb chunks → final QC
│   ├── 02_train_validation/ # Main/sub/taxonomy training
│   └── 03_inference/        # Holdout prediction, aggregation, metrics, and confusion matrices
├── data/
│   ├── manifests/           # Record provenance, labels, splits, and Mirus ANI/AF edges
│   ├── splits/              # 80/10/10 assignments by group
│   ├── preprocessed/        # Train/validation/inference data for main/sub/taxonomy
│   └── quarantine/          # Records excluded because of possible label contamination
├── runs/                    # Training checkpoints, best/last models, and loss plots
├── results/                 # Inference metrics, prediction tables, and confusion matrices
├── logs/
│   ├── preprocess/
│   ├── train/
│   └── inference/
```

### Overview files

| File | User action |
|---|---|
| `README.md` | Review the overall principles and result locations |
| `configs/data_sources.yaml` | Check or change original data paths |
| `configs/pipeline.yaml` | Check or change split, chunk, and training hyperparameters |

To run the pipeline, activate the conda environment and directly invoke the required Python scripts in the order of preprocessing, training, and inference.

### Responsibilities of internal scripts

| Code | Responsibility | Required? |
|---|---|---|
| `01_preprocessing/run_preprocessing.py` | Optional convenience tool that invokes all preprocessing stages | Optional |
| `01_build_manifest.py` | Build an inventory of IDs, labels, taxonomy, and hashes for all records | Required |
| `02_similarity.py` | Check Mirus skani ANI/AF and exact/MMseqs similarity across all sources before splitting | Required |
| `03_make_splits.py` | Assign entire groups to 80/10/10 splits and freeze the assignments | Required |
| `04_make_chunks.py` | Generate chunks of at most 5 kb within each split | Required |
| `05_validate_preprocessed.py` | Perform final checks for split leakage and main/sub consistency | Required |
| `06_prepare_taxonomy.py` | Generate taxonomy datasets for each rank | Optional |
| `02_train_validation/train.py` | Run and save main/sub/taxonomy training separately using the same training engine | Required |
| `03_inference/inference.py` | Evaluate one main or sub model on the holdout, with separate runs for each | Required |
| `03_inference/taxonomy.py` | Evaluate the selected taxonomy model | Optional |
| `check_environment.py` | Check conda packages, MMseqs, skani, and GPUs | Recommended before execution |
| `03_inference/evaluation.py` | Aggregate probabilities and calculate metrics and confusion matrices | Internal component called automatically during inference |

Preprocessing and training do not use a separate shared Python module. Helper functions for FASTA reading, checksums, logging, and model loading are included directly in the scripts that need them. However, `03_inference/evaluation.py` is an internal component containing only the aggregation and evaluation functions shared by main/sub inference and taxonomy evaluation.

## 2. Roles of the input datasets

| Source | Actual input | Model label | Default split unit | Important details |
|---|---|---:|---|---|
| ARC | `order_representatives_2026-09-03/fna/ARC__*.fna` | main 0 | genome | 172 files verified |
| BAC | `BAC__*.fna` in the same folder | main 0 | genome | 1,991 files verified |
| EUK | `EUK__*.fna` in the same folder | main 0 | genome | Genomes overlapping with EVE/PHAGE hosts are automatically quarantined |
| MITO | `organelle_host_balanced_2026-09-10/fna/MITO__*.fna` | main 0 | genome | 677 genomes, 77,598,909 bp. Mitochondrial snapshot balanced across host taxonomic groups |
| PLASTID | `PLASTID__*.fna` in the same folder | main 0 | genome | 258 genomes, 42,785,928 bp. Plastid snapshot balanced across host taxonomic groups |
| NCLDV | `gv-exports/NCLDV.fna` | main 1, sub 0 | **genus** | Falls back to genome if no usable genus is available |
| MIRUS | `gv-exports/MIRUS.fna` | main 1, sub 1 | **skani ANI 95% + AF 85% group** | All contigs are grouped by genome for comparison |
| EVE NCLDV | Records with `EVE-NCLDV__` in `eve_candidates_derep95.fna` | main 1, sub 0 | host assembly | EVE loci from the same host move together |
| EVE Mirus | Records with `EVE-MIRUS__` in `eve_candidates_derep95.fna` | main 1, sub 1 | host assembly | EVE loci from the same host move together |
| PHAGE | `metavr_phage_genera_all_quality_2026-09-04/fna` | main 1, sub 2 | vOTU | Snapshot of 2,612 uViGs already selected as the best single uViG per ICTV genus |

Any ARC/BAC/EUK/MITO/PLASTID genome overlapping with a host of EVE NCLDV, EVE Mirus, or PHAGE is automatically quarantined in preprocessing stage 1. Original `.fna` files are neither deleted nor moved. All contigs from the corresponding genome are excluded from training, validation, and inference, and the original path and supporting evidence are recorded in `data/quarantine/cellular_host_overlap.tsv`.

### How each dataset is read

- **FASTA files contain the DNA sequences.** A FASTA record is one sequence entry. One genome can contain several records when its sequence is divided into contigs.
- **TSV files describe those sequences.** They provide information such as genome names, taxonomy, sequence lengths, quality, and selection history. The pipeline links this information to the corresponding sequences using filenames or genome IDs.
- **All counts below describe the original inputs.** They are measured before quarantine, similarity filtering, splitting, and chunking. NCLDV/MIRUS/EVE counts were checked directly from FASTA files. Cellular/PHAGE counts come from snapshot metadata and agree with the raw manifest summary.
- **Input locations are configured in `configs/data_sources.yaml`.** Original FASTA files are read without modification.

#### ARC, BAC, and EUK

| Prefix | Original FASTA path | Genomes/files | FASTA records | Bases |
|---|---|---:|---:|---:|
| ARC | `fna/ARC__*.fna` | 172 | 9,460 | 302,682,872 |
| BAC | `fna/BAC__*.fna` | 1,991 | 225,723 | 6,131,933,134 |
| EUK | `fna/EUK__*.fna` | 130 | 1,366,361 | 7,677,494,753 |

- **What they are:** ARC contains archaeal genomes, BAC contains bacterial genomes, and EUK contains eukaryotic genomes.
- **Original directory:** `/home/fschulz/share/hyejong/genomes-training/order_representatives_2026-09-03/`.
- **Sequence files:** `fna/ARC__*.fna`, `fna/BAC__*.fna`, and `fna/EUK__*.fna`. Each file represents one genome and can contain several contigs.
- **Information file:** `metadata.tsv` contains 2,293 rows covering all three prefixes. The listed filenames match the FASTA files exactly.
- **How sequences are identified:** Headers such as `BAC__<genome_id>|<contig_id>` give the genome and contig IDs. The pipeline matches each filename to its metadata row and keeps all contigs from the same genome together during splitting.
- **What the metadata provides:** `assembly_accession` identifies the assembly, `organism_name` gives the organism, and `taxonomy_string` gives its lineage. Other fields describe assembly quality, representative status, sequence sizes, selection reasons, and download locations.
- **Where taxonomy comes from:** All three use sequences exported from GenBank. ARC and BAC use GTDB taxonomy and GTDB representative genomes. EUK uses NCBI Taxonomy.
- **How genomes were selected:** ARC and BAC are order representatives. EUK includes 120 order representatives, 5 additional Opisthokonta genomes, and 5 additional Viridiplantae genomes. Selection prioritizes representative/reference status, full genome status, assembly level, fewer contigs, genome length, and accession.
- **Important EUK detail:** 94 genomes were retained in full. The other 36 were already reduced to the first 100,000,000 bases when the original snapshot was created, with the final retained record trimmed if necessary. The fields `subsampled`, `subsample_target_bp`, and `subsampling_method` record this preparation.
- **Important taxonomy detail:** All rows contain a lineage string and an order name, but separate phylum, class, family, genus, and species fields are empty. The current code reads those separate fields and does not automatically extract missing ranks from the lineage string. Assembly accessions and organism names are still available for checking overlap with viral hosts.


#### MITO and PLASTID

| Prefix | Original FASTA path | Genomes/files | FASTA records | Bases |
|---|---|---:|---:|---:|
| MITO | `fna/MITO__*.fna` | 677 | 677 | 77,598,909 |
| PLASTID | `fna/PLASTID__*.fna` | 258 | 258 | 42,785,928 |
| Total | Both prefixes | 935 | 935 | 120,384,837 |

| Information file | Rows | Contents and use |
|---|---:|---|
| `metadata.tsv` | 935 | Final export inventory joined by `filename`, with accession filenames, organism/taxid, resolved taxonomy ranks, selection rules, input/output sizes, GC percentage, organelle type, and original database provenance |
| `source_metadata.tsv` | 935 | Selected original NCBI records, including accession/version, nuccore UID, title, organelle qualifier, topology, sequence length, organism/taxid, original FASTA path, download status, dates, and available sample/publication fields |
| `source_audit.tsv` | 17 | Manual source quality review with accession, action, reason, and evidence URL. Sixteen records were excluded and one retained with a caution |
| `selection.tsv` | 16,540 | Candidate inventory with selected status, exclusion reason, host group, original/resolved taxids, taxonomy ranks, order/family grouping keys, length, and fraction of bases outside ACGT |
| `provenance.json` | Not applicable | Source database path and checksums, exporter/taxonomy checksums, host caps, source exclusions, and exported genome/base totals |

| Host group in selected records | MITO | PLASTID | Total |
|---|---:|---:|---:|
| Metazoa | 100 | 0 | 100 |
| Viridiplantae | 100 | 100 | 200 |
| Fungi | 100 | 0 | 100 |
| Other groups | 377 | 158 | 535 |

- **What they are:** MITO contains mitochondrial genomes, and PLASTID contains plastid genomes, including chloroplast genomes.
- **Original directory:** `/home/fschulz/share/hyejong/genomes-training/organelle_host_balanced_2026-09-10/`.
- **Sequence files:** `fna/MITO__*.fna` and `fna/PLASTID__*.fna`. Each file contains one genome as one FASTA record.
- **Host balance:** The host counts in the table are selection categories, not model labels. The policy caps each of Metazoa, Viridiplantae, and Fungi at 100 per organelle type.
- **How sequences are identified:** FASTA headers begin with the original NCBI accession. The pipeline checks that this accession agrees with the filename, then creates an ID such as `MITO__<accession>`. All 935 filenames match the metadata inventory, and sequence lengths were previously verified against `output_fasta_bp`.
- **How genomes were selected:** The snapshot uses GenBank sequences and NCBI Taxonomy. Selection balances host orders and families, then favors fewer ambiguous bases and longer sequences. It also inherits an earlier reduction by host genus.
- **How to interpret quality:** `complete genome (source title)` means the original title describes the sequence as complete. It does not mean completeness was independently validated by this project. These 935 genomes are a selected snapshot, not all organelle genomes in GenBank.
- **Which paths are actually read:** The pipeline reads the exported FASTA files. Older FASTA and database paths in supporting metadata explain where the snapshot came from.


#### NCLDV and MIRUS

| Prefix | Original FASTA path | Genomes | FASTA records | Bases |
|---|---|---:|---:|---:|
| NCLDV | `/media/shared-expansion/db/gv-exports/NCLDV.fna` | 17,782 | 277,888 | 4,340,695,660 |
| MIRUS | `/media/shared-expansion/db/gv-exports/MIRUS.fna` | 1,677 | 33,987 | 434,193,248 |

| Information file | Rows | Contents and current pipeline use |
|---|---:|---|
| `labels.tsv` | 190,317 | Headerless file with two tab-separated fields: prefixed ID and a lineage separated by vertical bars. Read for authoritative exact ID taxonomy of NCLDV/MIRUS only |
| `old/genome_taxonomy.tsv` | 19,459 | One row per genome, covering 17,782 NCLDV and 1,677 MIRUS genomes. Fields include `genome`, `group`, taxonomy ranks, `taxonomy_majority`, and `taxonomy_strict`. Read as provisional fallback metadata |
| `old/taxonomy_lookup.tsv` | 311,875 | Record inventory covering the combined NCLDV/MIRUS contigs, with `header`, `gid`, `genome`, `group`, taxonomy ranks, `contig_len_bp`, majority taxonomy, and detailed assignment support. Inspected for documentation but not read by this pipeline |

- **What they are:** These are the conventional NCLDV and Mirus viral genome datasets, separate from EVE sequences inside host genomes.
- **Original directory:** `/media/shared-expansion/db/gv-exports/`.
- **NCLDV sequence file:** `/media/shared-expansion/db/gv-exports/NCLDV.fna`, containing 17,782 genomes, 277,888 records, and 4,340,695,660 bp.
- **MIRUS sequence file:** `/media/shared-expansion/db/gv-exports/MIRUS.fna`, containing 1,677 genomes, 33,987 records, and 434,193,248 bp.
- **How sequences are identified:** Each large FASTA holds contigs from many genomes. Headers such as `NCLDV__<genome_id>|<contig_id>` identify each genome and contig. The pipeline reads these IDs directly without permanently copying each genome into a separate FASTA file.
- **Exact taxonomy coverage:** Although `labels.tsv` contains 8,570 NCLDV keys and 3,013 MIRUS keys, only 8,025 NCLDV genomes and 1,473 MIRUS genomes match the current FASTA inputs exactly.
- **Other prefixes in `labels.tsv`:** BAC, ARC, PHAGE, PPV, MITO, PLASTID, EUK, and CRESS are also present. Their entries do not replace the snapshot metadata used by this pipeline or enable additional input sources.
- **How fallback taxonomy is used:** It helps group more NCLDV genomes by genus for splitting. Previously verified assignments place 15,069 NCLDV genomes into 2,734 usable genus groups. Genomes without a usable genus use genome and similarity grouping. Provisional taxonomy can appear in the manifest for either source, but is not accepted as a taxonomy training target.
- **How MIRUS is split:** MIRUS uses groups of similar genomes based on ANI and aligned fraction, rather than genus labels.


#### EVE NCLDV and EVE Mirus

| Configured source | FASTA prefix | Host grouping IDs | EVE records | Bases |
|---|---|---:|---:|---:|
| EVE_NCLDV | `EVE-NCLDV__` | 447 | 1,750 | 84,664,372 |
| EVE_MIRUS | `EVE-MIRUS__` | 81 | 312 | 13,997,657 |

- **What they are:** EVE sequences are viral sequence regions embedded in host genomes. One host assembly can contain several EVE regions.
- **Original sequence file:** `/media/shared-expansion/db/gv-exports/eve_candidates_derep95.fna`.
- **How the two types are selected:** `EVE_NCLDV` reads headers beginning with `EVE-NCLDV__`. `EVE_MIRUS` reads headers beginning with `EVE-MIRUS__`.
- **What the counts mean:** Host grouping IDs represent host assemblies rather than independent viral genomes.
- **Information files:** No dedicated EVE TSV was found in the export directory or its `old/` subdirectory. The nearby taxonomy TSVs do not provide an EVE metadata match used by this pipeline.
- **Where the identifying information comes from:** The FASTA header gives the host assembly and EVE region. Region annotations can include scaffold, start/end coordinates, tier, and confidence. The full original header is kept in the manifest.
- **How records are grouped:** The pipeline normalizes the host assembly ID and removes the `_eves` suffix. All EVE regions from the same host assembly stay in the same split.
- **How labels are assigned:** Main/sub labels come from the configured EVE source. Both enabled EVE sources have missing taxonomy in the current manifest, so they are used for main/sub classification but excluded from taxonomy training.
- **Disabled EVE_PPV data:** The same FASTA also contains 18,686 `EVE-PPV__` records, totaling 499,982,135 bp across 861 host grouping IDs. `EVE_PPV` is disabled in `configs/data_sources.yaml` while its subgroup role and possible label noise await review. These records are excluded from main and sub datasets.


#### PHAGE

| Recorded quality | uViGs |
|---|---:|
| Complete | 508 |
| High-quality | 1,635 |
| Medium-quality | 193 |
| Low-quality | 222 |
| Not-determined | 54 |
| Total | 2,612 |

- **What it is:** A selected viral snapshot containing one best uViG per assigned ICTV genus. A uViG is the viral genome entry used by this dataset.
- **Original directory:** `/home/fschulz/share/hyejong/genomes-training/metavr_phage_genera_all_quality_2026-09-04/`.
- **Sequence files:** `fna/PHAGE__*.fna`. There are 2,612 files, each with one uViG record, totaling 114,607,421 bp.
- **Information file:** `metadata.tsv` contains 2,612 rows, and its filenames match the FASTA files exactly.
- **How sequences are identified:** Headers use `PHAGE__<uvig>|record_1|img_contig_oid=<source_contig_id>`. The pipeline matches metadata by filename and uses `uvig` as the genome ID.
- **What the metadata provides:** Viral taxonomy, ICTV genus, vOTU, quality, completeness, contamination, genome type, topology, host taxonomy, host assignment method, sequence sizes, and source information.
- **How representatives were selected:** All rows have `selection_scope=genus_representative` and `selection_rank=1`. Selection prioritizes quality, completeness, lower contamination, length, and uViG ID. Full selected sequences are retained without subsampling.
- **How records are grouped for splitting:** Entries with the same `votu` stay together. A vOTU identifies a group of similar viral sequences. When it is missing, the pipeline uses the genome ID. Selection by ICTV genus and split grouping by vOTU serve different purposes.
- **Host distribution:** 1,117 entries have Bacteria hosts, 311 have Eukaryota hosts, 56 have Archaea hosts, and 1,128 have unspecified host taxonomy. Host assignment methods include VirusHostDB, iPHoP, isolate taxonomy, and vOTU.
- **How to interpret the PHAGE label:** The prefix does not mean that every entry is a bacteriophage. The sub model's PHAGE class represents this selected snapshot. It is not the whole MetaVR collection, and this holdout alone cannot establish performance on other virus types.
- **Which inputs are actually read:** The pipeline reads the existing FASTA files and `metadata.tsv`. The MetaVR database path in metadata records their origin. The pipeline does not read the MetaVR DuckDB or use MetaVR Other-virus candidates.

### Why a new FASTA file is not created for every prefix or genome

Explicit IDs such as `ARC__genome`, `EVE_NCLDV__host`, and `MIRUS__genome` are useful, so they are recorded in the manifest fields `source`, `genome_id`, `contig_id`, and `record_id`. However, copying large NCLDV/Mirus FASTA files into tens of thousands of individual genome files would increase inode usage and I/O costs. The pipeline therefore reads the originals directly and is designed so that **the manifest represents each record's file identity**.

Training data are also saved as one compressed CSV per split rather than one file per 5 kb chunk. This keeps the file count low while preserving the ability to trace each chunk to its prefix, genome, and contig.

## 3. Split design and leakage prevention

The critical order is as follows.

```text
Verify original records
  → Quarantine entire Cellular genomes overlapping with viral hosts
  → Define groups that must move together
  → Connect groups by exact or near similarity
  → Assign each entire group to train, validation, or inference
  → Only then generate 5 kb chunks
```

Randomly shuffling chunks before splitting can place nearly identical fragments from the same genome in both training and inference. A model could then achieve a high score by memorizing sequences it has already seen rather than generalizing biologically. This pipeline prevents that leakage by fixing the split first.

### Groups for each source

- ARC/BAC/EUK/MITO/PLASTID: All contigs from the same genome form one group. MITO/PLASTID have one record per file, but follow the same principle.
- NCLDV: When the taxonomic genus is clear, the entire genus forms one group. Inference therefore evaluates whether NCLDV recognition generalizes to genera unseen during training.
- Mirus: First combine all contigs from the same genome to calculate ANI and AF across whole genomes using skani. Connect genomes meeting both `ANI ≥95%` and `AF ≥85%` for the shorter genome, and place each entire connected component in the same split.
- EVE NCLDV/Mirus: All loci from the same host assembly belong to one split.
- PHAGE: Entries with the same `vOTU` stay together. A vOTU is an ID for a group of similar viral sequences. If no vOTU is available, use the genome ID for each `uvig`.

### Removing overlap between Cellular genomes and viral hosts

This check is performed before chunk generation.

- EVE NCLDV/EVE Mirus: Normalize host assembly accessions from EVE FASTA headers and accessions from Cellular metadata or filenames, then compare them by **exact accession**.
- PHAGE: Read species names from `host_taxonomy` in PHAGE metadata and compare them with `species_name` or `organism_name` in Cellular metadata by **exact species**.
- Matching only at genus level or having an empty name does not count as a match, to avoid excessive removal.
- If any match is found, all contigs in that Cellular genome file receive the quarantine reason `cellular_genome_is_viral_host`.

The current input snapshot contains 22 unique Cellular genomes meeting these criteria: 2 BAC, 19 EUK, and 1 PLASTID. EVE assembly matching identifies 18 unique genomes, some matching both EVE NCLDV and EVE Mirus. PHAGE host species matching identifies 4 genomes. These counts are automatically recalculated during preprocessing if the original snapshot changes.

Next, 5 kb audit fragments from all sources are compared using MMseqs2 Linclust. Fragments meeting at least 85% bidirectional coverage and 95% identity, or exact duplicates including reverse complements, connect their two groups into one similarity component. Functional testing confirmed that the installed MMseqs nucleotide Linclust recognizes the reverse complement strand, so each fragment is recorded only once. Mirus ANI/AF edges and these fragment edges are combined. If A connects to B and B connects to C, all three belong to the same split. If different labels occur within the same component, the corresponding records are quarantined rather than arbitrarily deciding which label is correct.

Final components are stratified by source and main label, then assigned deterministically to keep genome counts and total bases in each stratum close to 80/10/10. Row counts may therefore differ from exact 80/10/10 proportions. Preserving large groups such as NCLDV genera is more important than small deviations in split proportions.

### Safely splitting PHAGE (main 1, sub 2)

Read the 2,612 PHAGE `.fna` files that have already been selected. Records annotated with an NCLDV/Mirus phylum are excluded to prevent contamination of the sub model's PHAGE label, and entries with the same vOTU move together. Genome IDs are used when vOTUs are unavailable. Splitting then incorporates the results of exact and near similarity checks against other sources. There are no steps for querying the MetaVR database, extracting other viruses, or deduplicating PHAGE against MetaVR.

## 4. Preprocessing stages and generated files

**Run the following scripts manually in order.** `scripts/01_preprocessing/run_preprocessing.py` is an optional tool that runs the same stages together when desired.

### 4.1 `01_build_manifest.py`

Read every FASTA record and record the following.

- Original path/header, source, and genome/contig/host/vOTU IDs
- Main/sub labels and available taxonomy
- Length, N fraction, and canonical SHA-256 based on forward and reverse complement sequences
- Default split group and quarantine reasons
- Sizes and checksums of all input files

During this stage, Cellular genomes overlapping with EVE/PHAGE hosts are identified first, and `data/quarantine/cellular_host_overlap.tsv` is created. This table records the Cellular source, genome/file ID, original path, assembly accession, organism name, matching viral source, comparison method, and compared value. Quarantined genomes remain in the raw manifest for tracking, but have `split=quarantine` and are not included in chunk files.

For MITO/PLASTID, the file list in the original `metadata.tsv` is also compared with the FASTA file list. The presence and checksums of `source_metadata.tsv`, `source_audit.tsv`, `selection.tsv`, and `provenance.json` are recorded in the input inventory. Genome, record, and base counts per source are available in `data/manifests/source_summary.json`.

Bases are converted to uppercase, and `U→T` is applied. Ambiguous IUPAC characters such as `R/Y/S/W/K/M/B/D/H/V` are converted to `N`, but **N bases are not deleted**. Deleting Ns would change coordinates and lengths and join bases that were originally separated, potentially creating incorrect 5 kb fragments. Records with an N fraction above 20% are quarantined by default.

### 4.2 `02_similarity.py`

Perform the following two checks before splitting.

1. Combine all contigs of each conventional Mirus genome into a temporary FASTA file and compare every genome against every other genome using skani. Pairs meeting both `ANI ≥95%` and `AF ≥85%` based on the shorter genome are recorded in `data/manifests/mirus_genome_ani_edges.tsv`. Approximately 24% of current Mirus genomes have an N50 below 10 kb, so the skani `medium` preset is used.
2. Check exact hashes and MMseqs2 Linclust similarity at 95% identity and 85% bidirectional coverage across all sources. Connected component clustering is used instead of running `easy-search` comparisons of every pair among millions of fragments. Memory is limited to 48GB by default, and MMseqs internally partitions the input when needed. SHA-256 checks exhaustively detect identical sequences, including reverse complements. Near similarity candidate discovery uses Linclust for large datasets and is therefore approximate compared with exhaustive comparisons of all pairs.

Both results are merged into the final `similarity_edges.tsv`. Temporary FASTA files for skani and MMseqs audit fragments/databases are deleted after successful completion. Original FASTA files are not modified. Identity, coverage, threads, memory limits, the Linclust k-mer count, and the skani preset can be changed in the `similarity` section of `configs/pipeline.yaml`. On servers with limited memory, `memory_limit: 48G` can be lowered to a value such as `32G`, but more partitions will increase runtime.

If execution ends abruptly because of `Killed`, a server restart, or a similar event, large temporary directories may remain. The script stops without automatically deleting them. After checking the paths, explicitly specify `--clean-stale` as below to remove **only the derived intermediate files created by this stage** and restart.

```bash
python scripts/01_preprocessing/02_similarity.py --clean-stale
```

Progress is available in the main log and `data/preprocessed/similarity/linclust.log`. Proceed to the next stage only after the final `[DONE]` message appears and `data/manifests/similarity_audit.json` contains `complete: true`.

### 4.3 `03_make_splits.py`

Connect genome/genus/host/vOTU/similarity groups and create `data/manifests/genomes.tsv.gz` and `data/splits/group_assignments.tsv`. The seed, manifest checksum, and split settings are frozen in `frozen.json`.

### 4.4 `04_make_chunks.py`

Within each fixed split, cut each contig into chunks of at most 5,000 nt. Every base is used exactly once.

- 12,000 nt: `5000 + 5000 + 2000`
- 10,200 nt: Retain the final 200 nt and redistribute the last 5,200 nt into approximately `2600 + 2600`
- Original contig shorter than 500 nt: Retain it as one short chunk to preserve coordinates

No chunk exceeds 5 kb. If the final fragment is at least 500 nt, keep it as an additional chunk. If it is shorter than 500 nt, combine its span with the preceding 5 kb and redistribute that span evenly into two chunks rather than creating a fragment with too little signal. Bases are neither discarded nor overlapped.

During training only, each input is reverse complemented with a 50% probability on every presentation. This augmentation reduces sensitivity to sequence orientation because DNA can be read in either direction. It is not applied during validation or inference, ensuring reproducible results.

The generated files are as follows.

```text
data/preprocessed/main/{train,validation,inference}.csv.gz
data/preprocessed/sub/{train,validation,inference}.csv.gz
```

Sub data are not selected from main predictions. **All eligible records** whose ground truth sources are NCLDV, EVE NCLDV, Mirus, EVE Mirus, or PHAGE are taken directly from the same master holdout assignments. Main model outputs therefore do not affect selection of sub model inputs.

### 4.5 `05_validate_preprocessed.py`

Before training, check the following conditions using SQLite with storage on disk.

- Zero overlap of `split_group`, genomes, or contigs between splits
- Zero overlap of identical chunk hashes, including reverse complements, between splits
- Zero label conflicts for identical chunk hashes
- Exact agreement between ground truth class 1 chunks in main and all chunks in sub, including sequences, splits, and sub labels
- Current file checksums match the frozen manifest

Training cannot start if any check fails.

## 5. Main/Sub training and validation

Both models start from the same GenomeOcean base model, but are trained independently and can each be deployed separately.

```text
main dataset: Cellular(ARC/BAC/EUK/MITO/PLASTID) vs Viral(NCLDV/EVE NCLDV/Mirus/EVE Mirus/PHAGE)
sub dataset : NCLDV(NCLDV/EVE NCLDV) vs Mirus(Mirus/EVE Mirus) vs PHAGE(PHAGE)
```

Main and sub share the same original manifest and holdout assignments for leakage prevention, but **run independently with different data files and models**. Sub training and inference do not read main prediction files. Therefore, even a genome incorrectly predicted as Cellular by main remains in the sub holdout dataset if its ground truth source is viral.

### Loss

Main uses 2-class cross-entropy, and sub uses 3-class cross-entropy. This is a standard classification loss that penalizes the model to increase the probability of the correct class. Label smoothing of `0.05` reduces excessive certainty in a single class.

Long genomes could dominate training simply because they produce more 5 kb chunks. To prevent this, each chunk receives a weight of approximately `chunk length / total length of its genome`. Chunk weights for one genome sum to approximately 1. This is multiplied by a moderate `sqrt inverse class weight` to compensate for classes with fewer training genomes. Validation loss uses the same formula, but without gradient calculation or parameter updates.

### Default hyperparameters

All values can be changed in [configs/pipeline.yaml](configs/pipeline.yaml).
`training.common` defines shared defaults. Values with the same names under `training.main` or `training.sub` override those defaults only for the corresponding task. Learning rates are currently set separately to `3e-5` for main and `1e-5` for sub, and can be changed independently.

| Setting | Default | Meaning |
|---|---:|---|
| chunk size | 5,000 nt | DNA input length |
| max input tokens | 5,002 | Safe upper bound that does not allow tokenizer truncation |
| learning rate | main `3e-5`, sub `1e-5` | Initial fine-tuning rate for each task |
| optimizer | AdamW | Standard Transformer optimizer |
| weight decay | `0.01` | Reduces overfitting |
| scheduler | cosine | Reduces the learning rate later in training |
| warmup | 5% | Moderates abrupt updates at the start |
| train/validation batch | 8/8 per GPU | Reduce to 4 first if GPU memory runs out |
| gradient accumulation | 8 | Effective training batch of approximately 64 chunks |
| precision | BF16 | Verified on the current GPU |
| max epochs | main 10, sub 10 | Maximum before early stopping |
| validation/log | Every 0.1 epoch | Record train/validation loss and genome metrics |
| latest checkpoint | Every 0.5 epoch and at each epoch end | Overwrite the latest state available for resuming after interruption |
| early stopping | 3 epochs | Stop if validation genome macro-F1 does not improve |
| selection metric | genome macro-F1 | Criterion for selecting the `best` checkpoint |

To allow direct adjustment for each task, `learning_rate`, `warmup_ratio`, train/validation batch sizes, gradient accumulation, label smoothing, the latest checkpoint interval, early stopping, and class weight power are explicitly listed under both `main` and `sub`. Other `common` settings can also be added under a task using the same name to take precedence only for that task. Typos or unsupported names cause an error before training starts.

`history.csv` and `loss_curve.png` show training loss and validation loss on the same epoch axis. A log line has the format shown below.

When `--resume` is used, unsaved history rows beyond the saved epoch in `checkpoints/latest/trainer_state.json` are removed first. Traces of epochs moving backward during earlier resumptions are also cleaned up, and duplicate evaluations at the resume boundary epoch are not recorded twice in the CSV or plot.

Evaluating the full validation set every 0.1 epoch provides frequent loss measurements, but increases runtime compared with training alone. The recording interval can be adjusted through `log_and_validate_every_fraction_of_epoch`.

```text
[LOG] epoch 002.5/010 | gpu=0 | mem_peak=18.42 GiB | train_loss=0.4210 | val_loss=0.4872 | P=0.9012 | R=0.8875 | macro_F1=0.8938 | lr=2.10e-05
```

Start/end times, GPU index and type, and model/data/output paths are saved in `.log` files. Total runtime, peak CPU/GPU memory, input bytes, genome/contig/chunk/base counts, Mbp/s, and seconds/Mbp are saved in each stage's `runtime.json`.

### `best.pt` and `last.pt`

- `checkpoints/best/`: Atomically overwrite the model only when validation **genome-level macro-F1** improves. Keep the earlier model in a tie.
- `checkpoints/latest/`: Atomically overwrite model, optimizer, scheduler, RNG, and Trainer states every 0.5 epoch and at each epoch end. `--resume` uses only this directory.
- Standard `checkpoint-N` directories are not created, so checkpoints do not accumulate during training.
- `best.pt`: Atomically overwrite it together with `checkpoints/best/` whenever validation genome macro-F1 exceeds the previous best value. The best model up to that point therefore remains available even if training is interrupted. `last.pt` is exported from `checkpoints/latest/` after training finishes normally.

Because the final epoch may already be overfitted, `best.pt` is the appropriate model for final inference and deployment. Both training and inference load the tokenizer and base architecture from Hugging Face's `DOEJGI/GenomeOcean-100M-v1.2`. Inference applies the fine-tuned `best.pt` to that architecture. The final state in `last.pt` and `checkpoints/latest/` for continued training are also retained.

There is no need to download the model manually or create a `models/` folder in the project. On first access, `from_pretrained()` automatically downloads the model to the user's default Hugging Face cache and reuses that cache afterward. No model cache or symbolic link is created inside the project. For reproducibility, the repository's currently configured revision is pinned to `26dd3863d7e66a1e1d87ae4e45968bfdc1fa098f`.

## 6. Labeled inference and result interpretation

The inference 10% is used only at the end, after the model, hyperparameters, thresholds, and temperature have all been determined using validation. Choosing hyperparameters again after examining inference results effectively treats inference as validation and makes the final performance estimate optimistic.

Class probabilities are first saved for each 5 kb chunk. For contigs and genomes, log probabilities are averaged using `valid_length` as weights and then converted back to probabilities. Simple majority voting treats six chunks choosing a class with 51% confidence the same as four chunks choosing it with 99% confidence. Probability aggregation accounts for prediction strength and the lengths of short tail fragments.

The following are saved at the chunk, contig, and genome levels.

- confusion matrix: A table/figure showing where true and predicted classes are confused
- precision per class: The fraction of predictions assigned to that class that are correct
- recall per class: The fraction of actual members of that class that are detected. Particularly relevant when the goal is to avoid missing NCLDV
- F1 per class: The balance between precision and recall
- macro-F1: An equally weighted average of class F1 scores, preventing large classes from dominating the result
- balanced accuracy: The average recall across classes
- PR-AUC and Average Precision: Precision and recall performance as the threshold changes, useful for evaluating rare positives
- calibration/ECE: Check whether predictions reported with 90% probability are actually correct approximately 90% of the time
- genome-level split-group bootstrap 95% CI: An uncertainty interval obtained by resampling groups rather than chunks that are not independent

Main and sub are run separately during both training and inference. Main evaluates the two Cellular/Viral classes on the entire main inference split. Sub evaluates the three NCLDV/Mirus/PHAGE classes on its separate inference split containing NCLDV/EVE NCLDV/Mirus/EVE Mirus/PHAGE. No combined cascade evaluation that automatically passes main predictions to sub is performed, and no `end_to_end` results are generated.

## 7. Taxonomy is optional

After finalizing main/sub classification, classifiers for the `class`, `order`, `family`, and `genus` ranks can optionally be trained separately for NCLDV and Mirus.

- Use only authoritative exact taxonomy for conventional NCLDV/Mirus.
- EVEs without taxonomy are used for main/sub classification but are excluded from taxonomy loss.
- Only taxa with at least 2 training genomes become closed-set labels.
- If the highest probability is below the default threshold of 0.8, output `unknown` rather than forcing a taxonomy assignment.
- Because the NCLDV primary split keeps genera disjoint, all genus validation taxa may be unseen. In that case, the code stops rather than creating an invalid closed-set genus model. Starting with `order` or `family` is recommended.

Classification and taxonomy results are reported separately. Taxonomy performance does not change the primary main/sub scores.

## 8. Conda environment

PyTorch and GPU operations worked in the existing `GO` environment, but `transformers 4.51.3` and `huggingface-hub 1.24.0` were incompatible. A cloned environment named `GOholdout` was therefore created without modifying `GO`, and the following were verified.

- Python 3.11.15
- PyTorch 2.8.0+cu128, BF16, and CUDA GPU operations
- Transformers 4.51.3 + huggingface-hub 0.35.3
- MMseqs2 18.8cc5c
- skani 0.3.2
- `pip check`: No conflicts

The current environment is ready to use. To recreate it, run the following from the project directory.

```bash
conda create --name GOholdout --clone GO --yes
conda run -n GOholdout python -m pip install -r requirement.txt
conda install -n GOholdout -c conda-forge -c bioconda \
  mmseqs2 "skani>=0.3,<0.4" --yes
```

On a machine without an existing `GO` environment, create a Python 3.11 environment, install PyTorch 2.8.0 for CUDA 12.8 first, then install `requirement.txt`, MMseqs2, and skani in that order.

## 9. How to run

First enter the project directory and activate `GOholdout`. Then directly run preprocessing, main training, sub training, and inference separately when needed.

### 9.1 Prepare the project and environment

```bash
cd /mnt/taskmaster1/scratch/hyejong/02_gv_genomeocean_holdout
conda activate GOholdout
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --gpu-check
```

No separate model download command is required. The Hugging Face model is loaded automatically during the first training or inference run.

### 9.2 Preprocessing: manual execution by stage

```bash
python scripts/01_preprocessing/01_build_manifest.py
python scripts/01_preprocessing/02_similarity.py
python scripts/01_preprocessing/03_make_splits.py
python scripts/01_preprocessing/04_make_chunks.py
python scripts/01_preprocessing/05_validate_preprocessed.py
```

Mirus comparisons of all genome pairs with skani and MMseqs2 Linclust across all sources may take a long time because of the dataset size. Monitor progress in `logs/preprocess/` and `data/preprocessed/similarity/linclust.log`. Each stage does not automatically overwrite existing results. If derived temporary files remain from an interrupted stage 2, use `--clean-stale` as described above. There is no need to use `run_preprocessing.py` if you prefer manual execution.

### 9.3 Train main and sub

```bash
python scripts/02_train_validation/train.py \
  --task main \
  --run-name main_v1 \
  --gpu 0

python scripts/02_train_validation/train.py \
  --task sub \
  --run-name sub_v1 \
  --gpu 1
```

Because the models are independent, they can run on different GPUs. Initially, they can also be run sequentially to make logs and memory usage easier to monitor. To resume the same interrupted run, append `--resume` as follows.

```bash
python scripts/02_train_validation/train.py \
  --task main \
  --run-name main_v1 \
  --gpu 0 \
  --resume
```

### 9.4 Evaluate the final labeled inference 10%

Evaluate the main model.

```bash
python scripts/03_inference/inference.py \
  --task main \
  --run runs/main/main_v1 \
  --name holdout_v1 \
  --gpu 0
```

Evaluate the sub model separately.

```bash
python scripts/03_inference/inference.py \
  --task sub \
  --run runs/sub/sub_v1 \
  --name holdout_v1 \
  --gpu 0
```

Results are saved in `chunk/`, `contig/`, and `genome/` under `results/holdout_v1/main/` and `results/holdout_v1/sub/`, respectively. No `end_to_end/` directory is created. Existing outputs for the same result name and task combination are not overwritten.

### 9.5 Optional taxonomy

```bash
python scripts/01_preprocessing/06_prepare_taxonomy.py \
  --source NCLDV --rank family

python scripts/02_train_validation/train.py \
  --task taxonomy --source NCLDV --rank family \
  --run-name ncldv_family_v1 --gpu 0

python scripts/03_inference/taxonomy.py \
  --run runs/taxonomy/NCLDV/family/ncldv_family_v1 \
  --gpu 0
```

For Mirus, replace `NCLDV` with `MIRUS`.

## 10. What to change before execution

For the first baseline, keeping most default values is recommended. If changes are needed, review only these two files.

1. [configs/data_sources.yaml](configs/data_sources.yaml): When original file locations have changed
2. [configs/pipeline.yaml](configs/pipeline.yaml): Split, N criteria, chunks, MMseqs2, batch sizes, learning rates, epochs, and evaluation settings

Before running inference, first verify that `passed` is `true` in `data/preprocessed/qc_report.json` and that every source/class is present in all three splits in `data/splits/split_summary.json`. In particular, treat NCLDV genome-level recall and errors by NCLDV genus as primary results.