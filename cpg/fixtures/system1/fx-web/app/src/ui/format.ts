export class Formatter {
  render(total: number): string {
    return total.toFixed(2);
  }
}

export class FancyFormatter extends Formatter {
  render(total: number): string {
    return `** ${total.toFixed(2)} **`;
  }
}

function pick(): object {
  return new FancyFormatter();
}

export function formatTotal(total: number): string {
  return (pick() as Formatter).render(total);
}
