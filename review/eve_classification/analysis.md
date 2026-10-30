# EVE main1/sub1 holdout 분석

main: main1_10191118; sub: sub1_10191116. 두 모델은 독립적으로 평가했다.

정답률은 해당 source의 target-class recall과 같다. source만 선택하면 참 라벨이 한 종류이므로, 그 subset의 precision/F1을 EVE를 다른 source와 구별하는 성능으로 해석하지 않는다.

contig 표는 원본 FASTA 레코드(record_id), 즉 EVE locus별로 재집계한 값이다. chunk_id를 frozen manifest의 record_id·좌표에서 복원해 모든 예측에 한 번씩 대응시켰다. 집계는 기존 코드와 같은 valid_length 가중 평균 log-probability를 사용했다.

기존 평가 코드는 genome_id+contig_id로 묶는다. EVE 헤더를 |로 나눈 첫 contig 토큰만 사용하면서 서로 다른 좌위에 같은 contig_id가 생긴다. holdout에서 EVE_NCLDV 151개가 59개로, EVE_MIRUS 28개가 16개로 합쳐졌다. chunk 및 genome 결과는 기존 값을 사용했다.

EVE의 genome은 동일 host assembly 내 EVE들의 묶음이며 숙주 genome 전체의 예측이 아니다.

## main

| Source | 기준 | 평가 수 | 정답 수 | 정답률/Recall | 오분류 |
|---|---|---:|---:|---:|---|
| EVE_NCLDV | chunk | 1,990 | 1,913 | 96.13% | Cellular 77 |
| EVE_NCLDV | contig(EVE locus) | 151 | 134 | 88.74% | Cellular 17 |
| EVE_NCLDV | genome | 45 | 38 | 84.44% | Cellular 7 |
| EVE_MIRUS | chunk | 295 | 272 | 92.20% | Cellular 23 |
| EVE_MIRUS | contig(EVE locus) | 28 | 24 | 85.71% | Cellular 4 |
| EVE_MIRUS | genome | 10 | 9 | 90.00% | Cellular 1 |

## sub

| Source | 기준 | 평가 수 | 정답 수 | 정답률/Recall | 오분류 |
|---|---|---:|---:|---:|---|
| EVE_NCLDV | chunk | 1,990 | 1,913 | 96.13% | Mirus 51; PHAGE/Other Virus 26 |
| EVE_NCLDV | contig(EVE locus) | 151 | 141 | 93.38% | Mirus 4; PHAGE/Other Virus 6 |
| EVE_NCLDV | genome | 45 | 40 | 88.89% | Mirus 2; PHAGE/Other Virus 3 |
| EVE_MIRUS | chunk | 295 | 225 | 76.27% | NCLDV 65; PHAGE/Other Virus 5 |
| EVE_MIRUS | contig(EVE locus) | 28 | 19 | 67.86% | NCLDV 9; PHAGE/Other Virus 0 |
| EVE_MIRUS | genome | 10 | 5 | 50.00% | NCLDV 5; PHAGE/Other Virus 0 |

## 사용 EVE 데이터

아래는 전처리 후 자료이며 main/sub가 같은 EVE 레코드를 사용한다. 합산하지 않는다.

| Source | Split | Host assemblies | EVE loci | Chunks | Bases |
|---|---|---:|---:|---:|---:|
| EVE_NCLDV | train | 354 | 1,433 | 13,870 | 65,140,545 |
| EVE_NCLDV | validation | 44 | 135 | 1,644 | 7,862,960 |
| EVE_NCLDV | inference | 45 | 151 | 1,990 | 9,547,726 |
| EVE_NCLDV | total | 443 | 1,719 | 17,504 | 82,551,231 |
| EVE_MIRUS | train | 64 | 220 | 2,244 | 10,609,003 |
| EVE_MIRUS | validation | 7 | 31 | 197 | 914,731 |
| EVE_MIRUS | inference | 10 | 28 | 295 | 1,409,853 |
| EVE_MIRUS | total | 81 | 279 | 2,736 | 12,933,587 |

## 기존 저장 contig 결과와 비교

| Task | Source | 기존 contig 정답/수(정답률) | record_id별 재집계 정답/수(정답률) |
|---|---|---:|---:|
| main | EVE_NCLDV | 50/59 (84.75%) | 134/151 (88.74%) |
| main | EVE_MIRUS | 14/16 (87.50%) | 24/28 (85.71%) |
| sub | EVE_NCLDV | 54/59 (91.53%) | 141/151 (93.38%) |
| sub | EVE_MIRUS | 8/16 (50.00%) | 19/28 (67.86%) |

## 검증

기존 12개 source/level 결과는 prediction CSV와 metrics JSON이 모두 일치했다. 각 모델의 EVE holdout chunk 2,285개를 179개 원본 locus에 빠짐없이 한 번씩 대응시켰고, 모든 locus의 chunk 길이 합이 manifest 길이와 일치했다.

기존 데이터·평가 파일과 파이프라인 코드는 변경하지 않았다. 이 폴더에 분석 산출물만 생성했다.
