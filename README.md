# GenomeOcean holdout fine-tuning pipeline

이 프로젝트는 보유한 라벨 데이터 전체를 먼저 `train 80% / validation 10% / inference 10%`로 고정한 뒤, GenomeOcean main 모델과 sub 모델을 각각 fine-tuning하고 마지막 10%에서 평가한다.

- main: `0=Cellular`, `1=Viral`
  - Cellular: ARC, BAC, EUK, MITO, PLASTID
  - Viral: NCLDV, EVE NCLDV, Mirus, EVE Mirus, PHAGE
- sub: `0=NCLDV`, `1=Mirus`, `2=PHAGE`
  - NCLDV: 일반 NCLDV + EVE NCLDV
  - Mirus: 일반 Mirus + EVE Mirus
  - PHAGE: PHAGE snapshot
- taxonomy: primary classification이 끝난 뒤 선택적으로 수행하는 secondary task

- 원본 FASTA는 읽기 전용으로 사용한다. MetaVR DuckDB는 이 파이프라인에서 읽지 않는다.
- 심볼릭 링크를 만들지 않으며, 생성되는 파일은 모두 이 프로젝트 안에 저장한다. 
- 프로젝트 내부 경로는 스크립트 위치를 기준으로 계산하므로 폴더 전체를 다른 곳으로 옮겨도 작동한다. 
- 원본 데이터 경로만 `configs/data_sources.yaml`의 절대경로다. `data/preprocessed`는 사용자가 지정한 최종 전처리 데이터 위치다.
- NCLDV/Mirus/EVE에서 taxonomy 정보가 부족한 record도 main/sub 분류에는 사용할 수 있지만, taxonomy 학습 정답이 없는 record는 그 보조 학습에서 제외한다.

## 1. 단순화한 디렉터리와 역할

```text
02_gv_genomeocean_holdout/
├── README.md                 # 이 한 파일에 전체 설명 통합
├── requirement.txt          # Python package 버전
├── configs/
│   ├── data_sources.yaml    # 원본 절대경로, source별 label/group 규칙
│   └── pipeline.yaml        # 수정 가능한 모든 전처리·split·학습·평가 하이퍼파라미터들 설정
├── scripts/
│   ├── 01_preprocessing/    # manifest → 유사도 검사 → split → 5kb → 최종 QC
│   ├── 02_train_validation/ # main/sub/taxonomy 학습
│   ├── 03_inference/        # holdout 평가
│   └── 04_evaluation/       # 집계, metric, confusion matrix
├── data/
│   ├── manifests/           # record 출처·label·split과 Mirus ANI/AF edge
│   ├── splits/              # group별 80/10/10 배정
│   ├── preprocessed/        # main/sub/taxonomy의 train/validation/inference
│   └── quarantine/          # label 오염 가능성이 있어 사용하지 않은 목록
├── runs/                    # 학습 checkpoint, best/last, loss graph
├── results/                 # inference metric, 예측표, confusion matrix
├── logs/
│   ├── preprocess/
│   ├── train/
│   └── inference/
```


### 개요 파일들

| 파일 | 사용자가 하는 일 |
|---|---|
| `README.md` | 전체 원리와 결과 위치 확인 |
| `configs/data_sources.yaml` | 원본 경로 확인 또는 변경 |
| `configs/pipeline.yaml` | split·chunk·학습 hyperparameter 확인 또는 변경 |

실행할 때는 conda 환경을 활성화하고 필요한 Python 스크립트를 전처리, 학습, inference 순서로 직접 호출한다.

### 내부 스크립트가 하는 일

| 코드 | 역할 | 필수 여부 |
|---|---|---|
| `01_preprocessing/run_preprocessing.py` | 전처리 단계를 한 번에 호출하는 선택적 편의 도구 | 선택 |
| `01_build_manifest.py` | 모든 record의 ID·label·taxonomy·hash 목록 작성 | 필수 |
| `02_similarity.py` | split 전 Mirus skani ANI/AF와 전체 source exact/MMseqs 검사 | 필수 |
| `03_make_splits.py` | group 전체를 80/10/10에 배정하고 동결 | 필수 |
| `04_make_chunks.py` | split 안에서 최대 5 kb chunk 생성 | 필수 |
| `05_validate_preprocessed.py` | split leakage와 main/sub 일치 여부 최종 검사 | 필수 |
| `06_prepare_taxonomy.py` | rank별 taxonomy 데이터 생성 | 선택 |
| `02_train_validation/train.py` | 같은 학습 엔진으로 main/sub/taxonomy를 각각 별도 실행·저장 | 필수 |
| `03_inference/inference.py` | main 또는 sub 한 모델의 holdout 평가(각각 별도 실행) | 필수 |
| `03_inference/taxonomy.py` | 선택한 taxonomy 모델 평가 | 선택 |
| `check_environment.py` | conda package·MMseqs·skani·GPU 검사 | 실행 전 권장 |
| `04_evaluation/evaluation.py` | 확률 집계, metric, confusion matrix | 내부 부품 |

별도의 공통 Python 모듈은 사용하지 않는다. FASTA 읽기, checksum, 로그, 모델 로딩 같은 보조 함수는 필요한 실행 스크립트 안에 직접 들어 있다. 따라서 한 파일만 열어도 그 단계의 전체 동작을 확인할 수 있다. 대신 같은 함수가 여러 파일에 중복되므로, 이후 동작을 바꿀 때는 관련 스크립트를 모두 함께 수정해야 한다.


## 2. 입력 데이터의 역할

| source | 실제 입력 | 모델 label | 기본 split 단위 | 중요 사항 |
|---|---|---:|---|---|
| ARC | `order_representatives_2026-09-03/fna/ARC__*.fna` | main 0 | genome | 172개 파일을 확인함 |
| BAC | 같은 폴더의 `BAC__*.fna` | main 0 | genome | 1,991개 파일을 확인함 |
| EUK | 같은 폴더의 `EUK__*.fna` | main 0 | genome | EVE/PHAGE 숙주와 겹치는 genome은 자동 quarantine |
| MITO | `organelle_host_balanced_2026-09-10/fna/MITO__*.fna` | main 0 | genome | 677개, 77,598,909 bp; 숙주 분류군 균형을 맞춘 미토콘드리아 snapshot |
| PLASTID | 같은 폴더의 `PLASTID__*.fna` | main 0 | genome | 258개, 42,785,928 bp; 숙주 분류군 균형을 맞춘 색소체 snapshot |
| NCLDV | `gv-exports/NCLDV.fna` | main 1, sub 0 | **genus** | usable genus가 없으면 genome으로 fallback |
| MIRUS | `gv-exports/MIRUS.fna` | main 1, sub 1 | **skani ANI 95% + AF 85% group** | 모든 contig를 genome별로 묶어 비교 |
| EVE NCLDV | `eve_candidates_derep95.fna` 중 `EVE-NCLDV__` | main 1, sub 0 | host assembly | 같은 숙주의 EVE locus를 함께 이동 |
| EVE Mirus | `eve_candidates_derep95.fna` 중 `EVE-MIRUS__` | main 1, sub 1 | host assembly | 같은 숙주의 EVE locus를 함께 이동 |
| PHAGE | `metavr_phage_genera_all_quality_2026-09-04/fna` | main 1, sub 2 | vOTU | 이미 ICTV genus당 최선 uViG 1개로 선별된 2,612개 snapshot |

ARC/BAC/EUK/MITO/PLASTID 중 EVE NCLDV, EVE Mirus 또는 PHAGE의 숙주와 겹치는 genome은 전처리 1단계에서 자동 quarantine한다. 원본 `.fna`를 삭제하거나 이동하지 않으며, 해당 genome의 모든 contig를 학습·validation·inference에서 제외하고 `data/quarantine/cellular_host_overlap.tsv`에 원본 경로와 근거를 기록한다.

### 각 데이터를 어떻게 읽는가

`organelle_host_balanced_2026-09-10/`의 `metadata.tsv`와 `fna/`를 대조한 결과 총 **935개 파일=935개 genome**이고, 파일당 FASTA record는 하나다. 실제 서열 길이도 metadata의 `output_fasta_bp`와 935개 모두 일치한다. FASTA 헤더는 원본 accession으로 시작하고 파일명에는 `MITO__` 또는 `PLASTID__` prefix가 있으므로, 전처리는 **파일명과 헤더의 accession 일치**를 확인한 다음 `MITO__<accession>`처럼 genome ID를 만든다. 각 파일은 원본 그대로 읽고 복사·수정하지 않는다.

| 구분 | MITO | PLASTID | 합계 |
|---|---:|---:|---:|
| 선택된 genome | 677 | 258 | 935 |
| 염기 수 | 77,598,909 | 42,785,928 | 120,384,837 |
| Metazoa 숙주 | 100 | 0 | 100 |
| Viridiplantae 숙주 | 100 | 100 | 200 |
| Fungi 숙주 | 100 | 0 | 100 |
| 그 밖의 숙주 그룹 | 377 | 158 | 535 |

이 표의 숙주 그룹은 `selection.tsv`의 분류 기준이며, 바이러스/세포 label이 아니다. `source_metadata.tsv`는 선택된 원본 record 정보, `source_audit.tsv`는 원본 품질 검토(16개 제외, 1개 주의 후 유지), `selection.tsv`는 후보 16,540개 중 935개를 고른 내역, `provenance.json`은 원본과 선별 규칙을 설명한다. 이 보조 파일들도 입력 inventory에 기록해 재현성을 확인한다. 따라서 이 935개는 GenBank의 모든 organelle genome이 아니라 선별된 snapshot이며, 원본 title에 있는 “complete genome” 표기를 별도 실험으로 재검증한 것은 아니다.

`/home/fschulz/share/hyejong/genomes-training`의 Cellular/PHAGE snapshot에는 FASTA와 함께 제공된 `metadata.tsv`가 있으므로, 이 source들의 ID와 taxonomy는 해당 snapshot metadata를 기준으로 한다. `/media/shared-expansion/db/gv-exports/labels.tsv`는 NCLDV/Mirus taxonomy의 exact ID match에 사용한다.

실제 연결을 확인한 결과 NCLDV는 17,782 genomes/277,888 records 중 8,025 genomes, Mirus는 1,677 genomes/33,987 records 중 1,473 genomes가 `labels.tsv`와 exact match했다. 따라서 NCLDV genus split을 더 넓게 적용하기 위해 `old/genome_taxonomy.tsv`의 majority-derived genus를 **split grouping에만** 보조적으로 사용한다. authoritative exact label을 우선하고 보조 label을 fallback하면 NCLDV 15,069 genomes가 2,734 usable genus group에 묶이며, genus가 비어 있는 나머지는 genome+similarity group으로 fallback한다. 이 provisional taxonomy는 taxonomy 모델의 정답 label로는 사용하지 않는다.

PHAGE snapshot은 MetaVR의 phage 전체도 아니고 vOTU별 대표 집합도 아니다. 원본 snapshot의
`metadata.tsv`를 확인하면 2,612행 모두 `selection_scope=genus_representative`,
`selection_rank=1`이다. 즉, ICTV genus가 배정된 후보 중 quality, completeness, contamination,
길이 순으로 가장 좋은 uViG 하나를 미리 고른 데이터다. 모든 quality 단계가 후보에 포함되므로
최종 2,612개에는 Complete/HQ뿐 아니라 Medium/Low/Not-determined도 일부 있다. 각 선택된 uViG의
전체 서열은 자르거나 subsampling하지 않고 사용한다.

PHAGE snapshot은 MetaVR에서 선별된 기존 `.fna` 파일 집합이다. 과거 원본 metadata 대조에서 2,612개 uViG가 모두 MetaVR에 있었고, 이 중 2,143개가 Complete/HQ였다. **하지만 이 파이프라인은 MetaVR DuckDB나 MetaVR Other-virus 후보를 입력으로 사용하지 않는다.**

여기서 `PHAGE`는 파일 prefix일 뿐 순수 bacteriophage라는 뜻은 아니다. 이 snapshot에는
Bacteria host가 알려진 1,117개뿐 아니라 Eukaryota host 311개, Archaea host 56개와 host가
불명확한 1,128개도 들어 있다. 따라서 sub의 PHAGE class는 모든 바이러스의 대표 표본이 아니라
이 선별된 PHAGE snapshot에 한정된다. 다른 종류의 바이러스에 대한 일반화 성능은
이번 holdout만으로 주장할 수 없다.

### 왜 prefix/genome마다 새 FASTA 파일을 만들지 않는가

`ARC__genome`, `EVE_NCLDV__host`, `MIRUS__genome` 같은 명확한 ID는 매우 유용하므로 manifest의 `source`, `genome_id`, `contig_id`, `record_id`에 모두 기록한다. 그러나 NCLDV/Mirus 대형 FASTA를 genome별 파일 수만 개로 다시 복사하면 inode와 입출력 비용이 커진다. 따라서 원본을 그대로 읽고, **manifest가 각 record의 파일 역할을 대신하도록** 설계했다.

학습 데이터도 5 kb chunk별 파일을 만들지 않고 split마다 하나의 압축 CSV로 저장한다. 파일 수는 적지만 각 chunk가 어느 prefix/genome/contig에서 왔는지는 그대로 추적할 수 있다.

## 3. split과 leakage 방지 원리

가장 중요한 순서는 다음과 같다.

```text
원본 record 확인
  → 바이러스 숙주와 겹치는 Cellular genome 전체 quarantine
  → 함께 움직여야 할 group 정의
  → exact/near-similarity group 연결
  → group 전체를 train/validation/inference 중 하나에 배정
  → 그 후에만 5 kb chunk 생성
```

chunk를 먼저 무작위로 섞으면 같은 genome의 거의 동일한 조각이 train과 inference 양쪽에 들어갈 수 있다. 모델은 생물학적 일반화가 아니라 이미 본 서열을 기억해 높은 점수를 낼 수 있다. 이 pipeline은 split을 먼저 고정하므로 그런 누수를 막는다.

### source별 group

- ARC/BAC/EUK/MITO/PLASTID: 같은 genome의 모든 contig를 하나의 group으로 둔다. MITO/PLASTID는 파일당 record 하나지만 같은 원칙을 적용한다.
- NCLDV: taxonomy genus가 명확하면 같은 genus 전체를 하나로 묶는다. 따라서 inference에서는 학습 때 보지 않은 genus에 NCLDV 판별이 일반화되는지 평가한다.
- 일반 Mirus: 먼저 같은 genome의 모든 contig를 합쳐 skani로 genome 전체 ANI와 AF를 계산한다. `ANI ≥95%`이면서 짧은 genome의 `AF ≥85%`인 genome들을 연결하고, 연결 component 전체를 같은 split에 둔다.
- EVE NCLDV/Mirus: 같은 host assembly의 locus를 모두 한 split에 둔다.
- PHAGE: 같은 `vOTU`를 함께 둔다. vOTU는 서로 유사한 바이러스 서열 집단을 가리키는 ID다. vOTU가 없으면 `uvig`별 genome ID를 사용한다.

### Cellular–바이러스 숙주 겹침 제거

이 검사는 chunk를 만들기 전에 수행한다.

- EVE NCLDV/EVE Mirus: EVE FASTA header의 숙주 assembly accession과 Cellular metadata/파일명의 accession을 정규화한 뒤 **exact accession**으로 비교한다.
- PHAGE: PHAGE metadata의 `host_taxonomy`에서 species 이름을 읽고 Cellular metadata의 `species_name` 또는 `organism_name`과 **exact species**로 비교한다.
- genus만 같거나 이름이 비어 있는 경우는 과도한 제거를 피하기 위해 일치로 처리하지 않는다.
- 하나라도 일치하면 그 Cellular genome 파일의 모든 contig에 `cellular_genome_is_viral_host` quarantine 사유를 붙인다.

현재 입력 snapshot을 검사하면 고유 Cellular genome 22개가 해당한다. 구성은 BAC 2개, EUK 19개, PLASTID 1개다. EVE assembly 일치는 고유 genome 18개이며, 일부는 EVE NCLDV와 EVE Mirus 양쪽에 일치한다. PHAGE host species 일치는 4개다. 이 개수는 원본 snapshot이 바뀌면 전처리 실행 시 자동으로 다시 계산된다.

그 다음 모든 source의 5 kb audit fragment를 MMseqs2로 비교한다. 양방향 coverage 85% 이상, identity 95% 이상인 조각이나 exact/reverse-complement duplicate가 있으면 두 group을 하나의 similarity component로 연결한다. Mirus ANI/AF edge와 이 fragment edge를 모두 합치며, A-B, B-C가 연결되면 A-B-C 전체가 같은 split으로 간다. 서로 다른 label에서 이 정도로 유사한 서열이 나오면 어느 label이 맞는지 임의로 정하지 않고 양쪽 record를 quarantine한다.

최종 component를 source와 main label별로 나눈 뒤, 각 그룹의 genome 수와 총 염기 수가 80/10/10에 가깝도록 deterministic하게 배정한다. 그래서 정확히 row 80/10/10은 아닐 수 있다. NCLDV genus처럼 큰 그룹을 자르지 않는 것이 작은 비율 오차보다 더 중요하다.

### PHAGE(main 1, sub 2)를 안전하게 split하는 방법

이미 선별된 PHAGE `.fna` 2,612개를 읽는다. NCLDV/Mirus phylum으로 표시된 record는 sub의 PHAGE label 오염을 막기 위해 제외하고, 같은 vOTU는 함께 움직인다. vOTU가 없는 경우 genome ID를 사용한다. 그 후 다른 source와의 exact/near-similarity 검사 결과까지 합쳐 split한다. MetaVR DB 조회, 다른 바이러스 추출, PHAGE–MetaVR 중복 제거 단계는 없다.

## 4. 전처리 단계와 생성 파일

아래 코드를 **순서대로 수동 실행**한다. `scripts/01_preprocessing/run_preprocessing.py`는 원할 때만 같은 단계를 묶어 실행하는 선택적 도구다.

### 4.1 `01_build_manifest.py`

모든 FASTA record를 읽어 다음을 기록한다.

- 원본 path/header, source, genome/contig/host/vOTU ID
- main/sub label과 사용 가능한 taxonomy
- 길이, N 비율, forward 및 reverse-complement canonical SHA-256
- 기본 split group과 quarantine 이유
- 모든 입력 파일의 크기와 checksum

같은 단계에서 EVE/PHAGE 숙주와 겹치는 Cellular genome을 먼저 찾고 `data/quarantine/cellular_host_overlap.tsv`를 만든다. 이 표에는 Cellular source, genome/file ID, 원본 경로, assembly accession, organism 이름, 일치한 viral source, 비교 방법과 비교값이 기록된다. quarantine 대상 genome도 raw manifest에는 추적 목적으로 남지만 `split=quarantine`이므로 chunk 파일에는 들어가지 않는다.

MITO/PLASTID에서는 원본 `metadata.tsv`의 파일 목록과 FASTA 파일 목록도 비교한다. 그 외 `source_metadata.tsv`, `source_audit.tsv`, `selection.tsv`, `provenance.json`의 존재와 checksum도 입력 inventory에 남긴다. `data/manifests/source_summary.json`에서 source별 genome/record/염기 수를 볼 수 있다.

염기는 대문자로 바꾸고 `U→T`로 바꾼다. `R/Y/S/W/K/M/B/D/H/V` 같은 ambiguous IUPAC 문자는 `N`으로 바꾸지만 **N을 삭제하지 않는다**. N을 삭제하면 좌표와 길이가 달라지고 원래 떨어져 있던 염기가 붙으므로 잘못된 5 kb 조각이 생길 수 있다. N 비율이 20%를 넘는 record는 기본 quarantine한다.

### 4.2 `02_similarity.py`

split 전에 다음 두 검사를 수행한다.

1. 일반 Mirus의 모든 contig를 genome별 임시 FASTA로 묶고 skani로 all-vs-all 비교한다. `ANI ≥95%`와 짧은 genome 기준 `AF ≥85%`를 모두 통과한 쌍을 `data/manifests/mirus_genome_ani_edges.tsv`에 기록한다. 현재 Mirus 중 N50이 10 kb 미만인 genome이 약 24%여서 skani의 `medium` preset을 사용한다.
2. 모든 source에서 exact hash와 MMseqs2 95% identity/85% bidirectional coverage를 검사한다.

두 결과는 최종 `similarity_edges.tsv`에 합쳐진다. skani에 전달하기 위해 잠시 만드는 genome별 FASTA와 MMseqs audit fragment는 성공 후 삭제되며 원본 FASTA는 수정하지 않는다. threshold, thread, skani preset은 `configs/pipeline.yaml`의 `similarity` 부분에서 바꿀 수 있다.

### 4.3 `03_make_splits.py`

genome/genus/host/vOTU/similarity group을 연결해 `data/manifests/genomes.tsv.gz`와 `data/splits/group_assignments.tsv`를 만든다. seed, manifest checksum, split 설정은 `frozen.json`에 고정된다.

### 4.4 `04_make_chunks.py`

고정된 split 안에서 각 contig를 최대 5,000 nt로 자른다. 모든 염기는 정확히 한 번만 사용한다.

- 12,000 nt: `5000 + 5000 + 2000`
- 10,200 nt: 마지막 200 nt를 버리지 않고 마지막 5,200 nt를 약 `2600 + 2600`으로 재분배
- 원래 contig가 500 nt 미만: 좌표 보존을 위해 한 개의 짧은 chunk로 유지

즉, 5 kb보다 큰 chunk는 만들지 않는다. 끝 조각이 500 nt 이상이면 그대로 한 번 더 자른 결과로 남기고, 500 nt 미만이면 신호가 너무 약한 조각 하나를 만드는 대신 직전 5 kb와 합친 범위를 두 조각으로 균등 재분배한다. 염기를 버리거나 겹치게 하지 않는다.

train에서만 매번 50% 확률로 reverse complement를 입력한다. DNA는 양 방향으로 읽힐 수 있으므로 방향에 덜 민감하게 만드는 augmentation이다. validation/inference에는 적용하지 않아 결과를 항상 재현할 수 있게 한다.

생성 파일은 다음과 같다.

```text
data/preprocessed/main/{train,validation,inference}.csv.gz
data/preprocessed/sub/{train,validation,inference}.csv.gz
```

sub 데이터는 main 예측 결과에서 고르는 것이 아니다. 원래 정답 source가 NCLDV, EVE NCLDV, Mirus, EVE Mirus, PHAGE인 **모든 eligible record**를 같은 master holdout 배정에서 직접 가져온다. 즉 main이 어떤 결과를 내는지는 sub의 입력 선택에 영향을 주지 않는다.

### 4.5 `05_validate_preprocessed.py`

학습 전에 다음 조건을 SQLite를 사용해 disk-backed 방식으로 검사한다.

- split 사이 `split_group`, genome, contig overlap이 0
- split 사이 동일/reverse-complement chunk hash overlap이 0
- 동일 chunk hash의 label conflict가 0
- main의 정답 class 1 chunk와 sub의 전체 chunk가 sequence·split·sub label까지 완전히 동일
- 현재 파일 checksum이 frozen manifest와 동일

하나라도 실패하면 학습 코드가 시작되지 않는다.

## 5. Main/Sub train + validation

두 모델은 같은 GenomeOcean base model에서 시작하지만 서로 독립적으로 학습하고 각각 배포할 수 있다.

```text
main dataset: Cellular(ARC/BAC/EUK/MITO/PLASTID) vs Viral(NCLDV/EVE NCLDV/Mirus/EVE Mirus/PHAGE)
sub dataset : NCLDV(NCLDV/EVE NCLDV) vs Mirus(Mirus/EVE Mirus) vs PHAGE(PHAGE)
```

main과 sub는 같은 원본 manifest와 누수 방지용 holdout 배정을 공유하지만, **서로 다른 데이터 파일과 모델로 독립 실행**된다. sub 학습·inference는 main의 예측 파일을 읽지 않는다. 따라서 main에서 Viral을 Cellular로 잘못 예측한 genome도 정답 source가 바이러스라면 sub holdout 데이터에는 정상적으로 포함된다.

### Loss

main은 2-class cross-entropy, sub는 3-class cross-entropy를 쓴다. 쉽게 말해 정답 class의 확률이 높아지도록 벌점을 계산하는 일반적인 분류 loss다. label smoothing `0.05`를 적용해 모델이 한 class를 지나치게 100% 확신하는 것을 줄인다.

긴 genome은 5 kb chunk가 많다는 이유만으로 학습을 지배할 수 있다. 이를 막기 위해 각 chunk weight를 대략 `chunk 길이 / 그 genome의 전체 길이`로 둔다. 한 genome의 chunk weight 합은 약 1이 된다. 여기에 train genome 수가 적은 class를 보완하는 완만한 `sqrt inverse class weight`를 곱한다. validation loss에도 같은 수식을 사용하지만 gradient 계산과 parameter update는 하지 않는다.

### 기본 hyperparameter

모든 값은 [configs/pipeline.yaml](configs/pipeline.yaml) 한 곳에서 바꾼다.

| 설정 | 기본값 | 의미 |
|---|---:|---|
| chunk size | 5,000 nt | 모델 입력 DNA 길이 |
| max input tokens | 5,002 | tokenizer truncation을 허용하지 않는 안전 상한 |
| learning rate | `3e-5` | fine-tuning 시작값 |
| optimizer | AdamW | 일반적인 Transformer optimizer |
| weight decay | `0.01` | 과적합 완화 |
| scheduler | cosine | 학습 후반 learning rate 감소 |
| warmup | 5% | 초반 급격한 update 완화 |
| train/validation batch | GPU당 8/8 | GPU OOM이면 먼저 4로 낮춤 |
| gradient accumulation | 8 | 실질 train batch 약 64 chunks |
| precision | BF16 | 현재 GPU에서 확인 완료 |
| max epochs | main 10, sub 10 | early stopping 전 최대치 |
| validation/log | 0.5 epoch마다 | train/val loss와 genome metric 기록 |
| early stopping | 3 epochs | validation genome macro-F1 개선이 없으면 종료 |
| selection metric | genome macro-F1 | `best` checkpoint 선택 기준 |

`history.csv`와 `loss_curve.png`에는 같은 x축(epoch)에 train loss와 validation loss가 함께 그려진다. 로그 한 줄은 다음 형태다.

```text
[LOG] epoch 002.5/010 | gpu=0 | mem_peak=18.42 GiB | train_loss=0.4210 | val_loss=0.4872 | P=0.9012 | R=0.8875 | macro_F1=0.8938 | lr=2.10e-05
```

시작/완료 시간, GPU 번호·종류, 모델/data/output 경로는 `.log`에, 총 runtime, peak CPU/GPU memory, 입력 byte, genome/contig/chunk/base 수, Mbp/s와 seconds/Mbp는 각 단계의 `runtime.json`에 저장한다.

### `best.pt`와 `last.pt`

- `last.pt`: 학습이 종료된 마지막 상태다. 이어 학습할 때는 `runs/.../checkpoints/`의 마지막 checkpoint를 사용한다.
- `best.pt`: validation **genome-level macro-F1**이 가장 높았던 상태다. 동률이면 먼저 발견된 checkpoint를 유지한다.

마지막 epoch는 이미 과적합됐을 수 있으므로 최종 inference와 배포에는 `best.pt`를 쓰는 방식이 적절하다. 학습과 inference 모두 tokenizer와 base architecture를 Hugging Face의 `DOEJGI/GenomeOcean-100M-v1.2`에서 불러온다. inference에서는 그 architecture 위에 fine-tuning 결과인 `best.pt`를 적용한다. 마지막 상태인 `last.pt`와 이어 학습용 Trainer checkpoint도 함께 보관한다.

수동으로 모델을 다운로드하거나 프로젝트에 `models/` 폴더를 둘 필요는 없다. `from_pretrained()`가 최초 접근 때 Hugging Face의 사용자 기본 cache로 자동 다운로드하고 이후에는 cache를 재사용한다. 프로젝트 내부에는 모델 cache나 symlink를 만들지 않는다. 재현성을 위해 repository의 현재 설정 revision `26dd3863d7e66a1e1d87ae4e45968bfdc1fa098f`를 고정했다.

## 6. 라벨 있는 inference와 결과 해석

inference 10%는 모델, hyperparameter, threshold, temperature를 validation에서 모두 정한 뒤 마지막에 사용한다. inference 결과를 보고 다시 hyperparameter를 고르면 사실상 inference를 validation처럼 쓴 것이므로 최종 성능이 낙관적으로 변한다.

각 5 kb chunk의 class 확률을 먼저 저장하고, contig와 genome에서는 `valid_length`를 가중치로 log-probability를 평균한 뒤 다시 확률로 바꾼다. 단순 majority vote는 chunk 6개가 51%로 같은 class를 고른 경우와 chunk 4개가 99%로 고른 경우를 똑같이 취급한다. 확률 집계는 예측 강도와 짧은 tail 길이까지 반영한다.

각각 chunk, contig, genome 수준에서 다음을 저장한다.

- confusion matrix: 실제 class와 예측 class가 어디서 섞였는지 보여 주는 표/그림
- class별 precision: 그 class라고 예측한 것 중 맞은 비율
- class별 recall: 실제 그 class 중 찾아낸 비율. NCLDV를 놓치지 않는 목표에는 특히 중요
- class별 F1: precision과 recall의 균형
- macro-F1: 각 class F1을 동일 비중으로 평균. 큰 class가 결과를 독점하지 않음
- balanced accuracy: 각 class recall의 평균
- PR-AUC와 Average Precision: threshold를 바꾸었을 때 precision-recall 성능. 희귀 positive 평가에 유용
- calibration/ECE: 90% 확률이라고 말한 예측이 실제로도 약 90% 맞는지 확인
- genome-level split-group bootstrap 95% CI: 서로 독립적이지 않은 chunk 대신 group을 재표집한 불확실성 범위

main과 sub는 학습뿐 아니라 inference도 각각 따로 실행한다. main은 전체 main inference split에서
Cellular/Viral 2개 class를 평가한다. sub는 NCLDV/EVE NCLDV/Mirus/EVE Mirus/PHAGE가 들어 있는
별도의 sub inference split에서 NCLDV/Mirus/PHAGE 3개 class를 평가한다. main 예측을 sub로 자동
전달하는 결합(cascade) 평가와 `end_to_end` 결과는 만들지 않는다.

## 7. Taxonomy는 선택 사항

먼저 main/sub classification을 확정한 후 NCLDV와 Mirus 각각에 대해 `class`, `order`, `family`, `genus` rank classifier를 선택적으로 학습한다.

- 일반 NCLDV/Mirus의 authoritative exact taxonomy만 사용한다.
- taxonomy가 없는 EVE는 main/sub 분류에는 쓰지만 taxonomy loss에는 넣지 않는다.
- train에서 최소 2 genomes가 있는 taxon만 closed-set label로 만든다.
- 최고 확률이 기본 0.8 미만이면 억지 taxonomy 대신 `unknown`을 출력한다.
- NCLDV primary split이 genus-disjoint이므로 genus validation taxon이 모두 unseen일 수 있다. 이때 코드는 잘못된 closed-set genus 모델을 만들지 않고 중단한다. 처음에는 `order` 또는 `family`부터 권장한다.

classification 결과와 taxonomy 결과는 별도로 보고한다. taxonomy 성능이 primary main/sub 점수를 바꾸지 않는다.

## 8. Conda 환경

기존 `GO`는 PyTorch/GPU는 작동했지만 `transformers 4.51.3`과 `huggingface-hub 1.24.0`이 호환되지 않았다. 그래서 `GO`를 건드리지 않고 복제한 `GOholdout`을 만들었으며 다음을 확인했다.

- Python 3.11.15
- PyTorch 2.8.0+cu128, BF16, CUDA GPU 연산
- Transformers 4.51.3 + huggingface-hub 0.35.3
- MMseqs2 18.8cc5c
- skani 0.3.2
- `pip check`: 충돌 없음

현재 환경은 바로 사용할 수 있다. 다시 만들 때는 프로젝트에서 다음을 실행한다.

```bash
conda create --name GOholdout --clone GO --yes
conda run -n GOholdout python -m pip install -r requirement.txt
conda install -n GOholdout -c conda-forge -c bioconda \
  mmseqs2 "skani>=0.3,<0.4" --yes
```

기존 `GO`가 전혀 없는 머신이라면 Python 3.11 환경을 만든 뒤 CUDA 12.8용 PyTorch 2.8.0을 먼저 설치하고 `requirement.txt`, MMseqs2와 skani 순으로 설치한다.

## 9. 실행 방법

먼저 프로젝트로 이동하고 `GOholdout` 환경을 활성화한다. 이후 전처리, main 학습, sub 학습, inference를 필요한 시점에 각각 직접 실행한다.

### 9.1 프로젝트와 환경 준비

```bash
cd /mnt/taskmaster1/scratch/hyejong/02_gv_genomeocean_holdout
conda activate GOholdout
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --gpu-check
```

별도 모델 다운로드 명령은 없다. 첫 학습 또는 inference에서 Hugging Face 모델을 자동으로 불러온다.

### 9.2 전처리: 단계별 수동 실행

```bash
python scripts/01_preprocessing/01_build_manifest.py
python scripts/01_preprocessing/02_similarity.py
python scripts/01_preprocessing/03_make_splits.py
python scripts/01_preprocessing/04_make_chunks.py
python scripts/01_preprocessing/05_validate_preprocessed.py
```

Mirus skani all-vs-all과 전체 MMseqs2 비교는 데이터가 크므로 오래 걸릴 수 있다. 진행 상황은 `logs/preprocess/`에서 확인한다. 각 단계는 기존 결과를 자동으로 덮어쓰지 않는다. `run_preprocessing.py`는 수동 실행을 원한다면 사용할 필요가 없다.

### 9.3 main과 sub 학습

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

두 모델은 독립적이므로 다른 GPU에서 각각 실행할 수 있다. 처음에는 로그와 memory 사용량을 확인하기 쉽도록 순서대로 실행해도 된다. 중단된 동일 run을 이어갈 때는 다음처럼 마지막에 `--resume`을 붙인다.

```bash
python scripts/02_train_validation/train.py \
  --task main \
  --run-name main_v1 \
  --gpu 0 \
  --resume
```

### 9.4 라벨 있는 최종 inference 10% 평가

main 모델을 평가한다.

```bash
python scripts/03_inference/inference.py \
  --task main \
  --run runs/main/main_v1 \
  --name holdout_v1 \
  --gpu 0
```

sub 모델은 별도로 평가한다.

```bash
python scripts/03_inference/inference.py \
  --task sub \
  --run runs/sub/sub_v1 \
  --name holdout_v1 \
  --gpu 0
```

결과는 각각 `results/holdout_v1/main/`과 `results/holdout_v1/sub/` 아래의
`chunk/`, `contig/`, `genome/`에 저장된다. `end_to_end/`는 만들지 않는다.
같은 결과 이름과 같은 task 조합은 덮어쓰지 않는다.

### 9.5 선택적 taxonomy

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

Mirus는 `NCLDV`를 `MIRUS`로 바꾸면 된다.

## 10. 실행 전에 수정할 곳

대부분은 기본값 그대로 첫 baseline을 실행하는 것을 권장한다. 수정이 필요하면 두 파일만 확인한다.

1. [configs/data_sources.yaml](configs/data_sources.yaml): 원본 파일 위치가 바뀐 경우
2. [configs/pipeline.yaml](configs/pipeline.yaml): split, N 기준, chunk, MMseqs2, batch, learning rate, epoch, 평가 설정

inference를 실행하기 전에 `data/preprocessed/qc_report.json`의 `passed`가 `true`인지, `data/splits/split_summary.json`에서 각 source/class가 세 split에 존재하는지 먼저 확인한다. 특히 NCLDV genome-level recall과 NCLDV genus별 오류를 primary 결과로 본다.
