async function loadAll(ids) {
  const results = [];
  ids.forEach(async (id) => {
    const data = await fetchItem(id);
    results.push(data);
  });
  return results;
}

module.exports = { loadAll };
