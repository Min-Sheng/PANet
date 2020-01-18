import json

cls_id_table = {
  # Fold-0
  '1': 'BBBC039', 
  '2': 'Hela', 
  # Fold-1
  '3': 'MicroNet', 
  '4': 'BBBC007', 
  # Fold-2
  '5': 'MoNuSeg', 
  '6': 'BBBC020_cell', 
  '7': 'BBBC020_nuclei', 
  # Fold-3
  '8': 'NucleiSeg', 
  '9': 'ClusterNuclei',
  # Fold-4
  '10': 'TNBC', 
  '11': 'ISBI2009',
  # Fold-5
  '12': 'BBBC018_cell', 
  '13': 'BBBC018_nuclei'
}
with open('/tmp2/PANet/FSSCell_cls_id.json', 'w') as fp:
    json.dump(cls_id_table, fp, indent=4)