var utterances = ['a', 'b', 'c']
var meanings = [1, 2, 3]
var meaningPrior = function() {
  return categorical({vs: meanings, ps: [0.1, 0.3, 0.6]})
}
var utterancePrior = function() {
  return categorical({vs: utterances, ps: [0.1, 0.3, 0.9]})
}


var semantics = function(u, m) {
  var mapper = {
    'a': [1], 
    'b': [1, 2],
    'c': [3],
  }
  return mapper[u].includes(m)
}


var L0 = function(u) {
  Infer({method: 'enumerate'}, function() {
    var m = meaningPrior()
    factor(semantics(u, m) ? 0 : -Infinity)
    return m
  })
}

var S1 = function(m) {
  Infer({method:'enumerate'}, function() {
    var u = utterancePrior()
    var L = L0(u)
    factor(L.score(m))
    return u
  })
}

var L1 = function(u) {
  Infer({method: 'enumerate'}, function() {
    var m = meaningPrior()
    var S = S1(m)
    factor(S.score(u))
    return m
  })
}

console.log(S1(1))
console.log(S1(2))
console.log(S1(3))
console.log(Math.exp(L0('a').score(1)))
console.log(Math.exp(L0('b').score(1)))
console.log(Math.exp(L0('b').score(2)))
console.log('====')
console.log(Math.exp(L1('a').score(1)))
console.log(Math.exp(L1('b').score(1)))
console.log(Math.exp(L1('b').score(2)))