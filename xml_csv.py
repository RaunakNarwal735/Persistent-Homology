import xml.etree.ElementTree as ET
import csv
import os

def sbml_to_csv(sbml_file, out_prefix=None):
    tree = ET.parse(sbml_file)
    root = tree.getroot()
    ns = {'sbml': 'http://www.sbml.org/sbml/level2/version4'}
    if out_prefix is None:
        out_prefix = os.path.splitext(sbml_file)[0]

    species = []
    for s in root.findall('.//sbml:listOfSpecies/sbml:species', ns):
        species.append([
            s.get('id'),
            s.get('name'),
            s.get('initialConcentration') or s.get('initialAmount') or ''
        ])
    with open(out_prefix + "_species.csv", "w", newline='') as f:
        csv.writer(f).writerows([["id", "name", "initial"]] + species)

    reactions = []
    for r in root.findall('.//sbml:listOfReactions/sbml:reaction', ns):
        rid = r.get('id')
        reactants = [reac.get('species') for reac in r.findall('./sbml:listOfReactants/sbml:speciesReference', ns)]
        products = [prod.get('species') for prod in r.findall('./sbml:listOfProducts/sbml:speciesReference', ns)]
        reactions.append([rid, ','.join(reactants), ','.join(products)])
    with open(out_prefix + "_reactions.csv", "w", newline='') as f:
        csv.writer(f).writerows([["id", "reactants", "products"]] + reactions)

if __name__ == "__main__":
    import sys
    sbml_to_csv(sys.argv[1])
