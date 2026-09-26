import { expect, test } from 'bun:test';
import { PDFDocument, PDFName, PDFString } from 'pdf-lib';
import { checkFile } from './validation';

test('PDF validation accepts command-like text inside content streams and metadata', async () => {
  const pdf = await PDFDocument.create();
  const page = pdf.addPage();
  pdf.setTitle('Static reference to /JavaScript and /AA');
  page.node.set(PDFName.of('Contents'), pdf.context.register(pdf.context.stream('% /AA /JS /OpenAction /URI are plain comment bytes\n')));
  const bytes = await pdf.save({ useObjectStreams: false });
  expect(Buffer.from(bytes).toString('latin1')).toContain('/AA');
  await checkFile('pattern.pdf', bytes, 'application/pdf');
});

test('PDF validation rejects active dictionaries, including compressed indirect objects', async () => {
  for (const compressed of [false, true]) {
    const pdf = await PDFDocument.create(); pdf.addPage();
    const action = pdf.context.register(pdf.context.obj({ S: PDFName.of('JavaScript'), JS: PDFString.of('app.alert("test")') }));
    pdf.catalog.set(PDFName.of('OpenAction'), action);
    await expect(checkFile('pattern.pdf', await pdf.save({ useObjectStreams: compressed }), 'application/pdf')).rejects.toThrow('Active PDF');
  }
});

test('PDF validation decodes escaped names and rejects malformed PDFs', async () => {
  const objects = [
    '<< /Type /Catalog /Pages 2 0 R /#4FpenAction 4 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>',
    '<< /S /Java#53cript /J#53 (app.alert\("test"\)) >>',
  ];
  let text = '%PDF-1.7\n'; const offsets = [0];
  objects.forEach((object, index) => { offsets.push(text.length); text += `${index + 1} 0 obj\n${object}\nendobj\n`; });
  const xref = text.length;
  text += `xref\n0 5\n0000000000 65535 f \n${offsets.slice(1).map(offset => `${String(offset).padStart(10, '0')} 00000 n \n`).join('')}trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  await expect(checkFile('pattern.pdf', Buffer.from(text), 'application/pdf')).rejects.toThrow('Active PDF');
  // Same-length substitution preserves every xref offset; only the escaped
  // catalog key is active, without a second forbidden action name to catch it.
  const lowerCase = text.replace('/#4FpenAction', '/#4fpenAction').replace('/Java#53cript', '/Noop#53cript').replace('/J#53 ', '/N#53 ');
  await expect(checkFile('pattern.pdf', Buffer.from(lowerCase), 'application/pdf')).rejects.toThrow('Active PDF');
  await expect(checkFile('pattern.pdf', Buffer.from('%PDF-1.7\ninvalid'), 'application/pdf')).rejects.toThrow('Invalid PDF');
});
