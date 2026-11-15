set -e 

mkdir -p data
cd data

echo 'downloading CREB1 peak file from GEO'
wget -c ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE127nnn/GSE127408/suppl/GSE127408_ENCFF550TXR_optimal_idr_thresholded_peaks_GRCh38.bed.gz
gunzip -f GSE127408_ENCFF550TXR_optimal_idr_thresholded_peaks_GRCh38.bed.gz
mv GSE127408_ENCFF550TXR_optimal_idr_thresholded_peaks_GRCh38.bed creb1_peaks.bed
echo 'peak file downloaded and extracted'

if [ ! -f 'Homo_sapiens.GRCh38.dna.primary_assembly.fa' ] && [ ! -f 'Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz' ]; then
    wget -c ftp://ftp.ensembl.org/pub/release-113/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
    gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
else
    echo 'genome file already exists, skipping download'
fi

echo 'all data downloaded succesfully'
cd ..