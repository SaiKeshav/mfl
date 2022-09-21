# Download IndicLink test data
cd ../data
git clone https://github.com/google-research-datasets/IndicLink

cd IndicLink
cat WikiData_facts.jsonl.part-* > WikiData_facts.jsonl
cp WikiData_facts.jsonl IndicLink_release.jsonl ..

rm -r IndicLink