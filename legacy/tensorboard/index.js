function filterNames() {
  // Get value of input
  let filterValue = document.getElementById('filterInput').value.toUpperCase();
  // Get pokemon ul
  let ul = document.getElementById('runs');
  // Get li from ul
  let li = ul.querySelectorAll('li.collection-item');
  // Loop through collection-item li
  for(let i = 0; i < li.length; i++){
    // let a = li[i].getElementsByTagName('a')[0];
    // if matched
    if(li[i].innerHTML.toUpperCase().indexOf(filterValue) > -1) {
        li[i].style.display = '';
    } else {
        li[i].style.display = 'none';
    }
  }
}

function shiftTensorboard(e) {
    document.getElementById('tensorboard').setAttribute(
        'src',
        'http://localhost:' + e.target.id
    );
    let current = document.getElementsByClassName('orange').item(0)
    if (current) {
        current.className = 'collection-item';
    }
    e.target.className += ' orange accent-2'
}

document.getElementById('filterInput').addEventListener('keyup', filterNames);

let items = document.getElementsByClassName('collection-item')

for (i = 0; i < items.length; i++) {
    console.log(items.item(i))
    items.item(i).onclick = shiftTensorboard;    
}

