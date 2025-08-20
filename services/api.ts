import type { ExtractionResult, TableData, HtmlResult } from '../types';
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from "@google/genai";

declare const pdfjsLib: any;

pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.5.136/pdf.worker.min.mjs`;

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
   {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
   {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];


const structuredResponseSchema = {
    type: Type.OBJECT,
    properties: {
        tables: {
            type: Type.ARRAY,
            description: "List of all tables parsed from the provided HTML.",
            items: {
                type: Type.OBJECT,
                properties: {
                    title: { type: Type.STRING, description: "The title or a brief summary of the table's content." },
                    columns: { type: Type.ARRAY, items: { type: Type.STRING }, description: "The column headers of the table." },
                    rawData: { type: Type.ARRAY, items: { type: Type.ARRAY, items: { type: Type.STRING } }, description: "The table data as a 2D array of strings, row by row." },
                    pageNumber: { type: Type.NUMBER, description: "The original page number this table was extracted from." },
                },
                required: ["title", "columns", "rawData", "pageNumber"]
            }
        },
        metadata: {
            type: Type.OBJECT,
            properties: {
                currency: { type: Type.STRING, description: "The main currency mentioned in the tables (e.g., 'NIS', 'USD', 'אלפי ש\"ח'). Default to 'לא צוין' if not found." },
                reportingPeriod: { type: Type.STRING, description: "The main reporting period of the document (e.g., 'ליום 31 בדצמבר 2022'). Default to 'לא צוין' if not found." }
            },
             required: ["currency", "reportingPeriod"]
        }
    },
    required: ["tables", "metadata"]
};


const canvasToBase64 = (canvas: HTMLCanvasElement): string => {
    return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
};

const arrayToCsv = (header: string[], data: string[][]): string => {
    const csvRows = [];
    const escapedHeader = header.map(cell => `"${String(cell || '').replace(/"/g, '""')}"`).join(',');
    csvRows.push(escapedHeader);

    for (const row of data) {
        const escapedRow = row.map(cell => `"${String(cell || '').replace(/"/g, '""')}"`).join(',');
        csvRows.push(escapedRow);
    }
    return csvRows.join('\n');
};

/**
 * Step 1: Processes a PDF file page by page, converting each page to an image and extracting table structures as HTML.
 */
export const extractHtmlFromPdf = async (file: File, onProgress: (progress: { current: number, total: number }) => void): Promise<HtmlResult[]> => {
    const allHtmlResults: HtmlResult[] = [];
    const arrayBuffer = await file.arrayBuffer();
    const pdfDoc = await pdfjsLib.getDocument(arrayBuffer).promise;
    const numPages = pdfDoc.numPages;

    for (let i = 1; i <= numPages; i++) {
        onProgress({ current: i, total: numPages });
        const page = await pdfDoc.getPage(i);
        const viewport = page.getViewport({ scale: 1.5 });

        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (!context) continue;
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        await page.render({ canvasContext: context, viewport: viewport }).promise;
        const base64Data = canvasToBase64(canvas);

        const imagePart = { inlineData: { mimeType: 'image/jpeg', data: base64Data } };
        
        try {
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: { parts: [
                    {text: "From the provided image of a document page, extract ALL tables into clean, semantic HTML `<table>` elements. Preserve the original text and structure, including headers and rows. If no tables are found on the page, return an empty string."},
                    imagePart
                ]},
                config: {
                    temperature: 0.1,
                    safetySettings,
                },
            });

            const htmlContent = response.text.trim();
            if (htmlContent) {
                // Find all table tags
                const tables = htmlContent.match(/<table[\s\S]*?<\/table>/g);
                if(tables) {
                    tables.forEach(tableHtml => {
                         allHtmlResults.push({ pageNumber: i, html: tableHtml });
                    });
                }
            }
        } catch (error) {
            console.error(`Error processing page ${i}:`, error);
            // Optionally, decide if you want to stop or continue on error
        }
    }

    return allHtmlResults;
};

/**
 * Step 2: Takes the extracted HTML tables and structures them into a final JSON object.
 */
export const structureDataFromHtml = async (htmlResults: HtmlResult[], documentName: string): Promise<ExtractionResult> => {

    const htmlInput = htmlResults.map(r => `<!-- Page ${r.pageNumber} -->\n${r.html}`).join('\n\n');

    const prompt = `Please analyze the following HTML tables extracted from a multi-page financial document. Structure the information into a single JSON object according to the provided schema. Infer the overall document metadata (currency, reporting period) from the content. It is critical that the 'pageNumber' for each table in the output corresponds to the source page number indicated in the HTML comments (e.g., <!-- Page 4 -->).

${htmlInput}`;


    const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
        config: {
            responseMimeType: 'application/json',
            responseSchema: structuredResponseSchema,
            safetySettings,
        },
    });

    const parsedJson = JSON.parse(response.text);

    const tables: TableData[] = (parsedJson.tables || []).map((table: any) => ({
        ...table,
        confidence: 0.98,
        errors: [],
        html: table.html || `<table>...</table>`, // Placeholder as we already have it
        csv: arrayToCsv(table.columns, table.rawData),
    }));

     // Re-inject the original HTML back into the final structured object for display
    tables.forEach(table => {
        const originalHtml = htmlResults.find(hr => hr.pageNumber === table.pageNumber);
        if (originalHtml) {
            table.html = originalHtml.html;
        }
    });


    return {
        documentName: documentName,
        totalPages: Math.max(...htmlResults.map(r => r.pageNumber)),
        tables: tables,
        metadata: {
            currency: parsedJson.metadata?.currency || 'לא צוין',
            reportingPeriod: parsedJson.metadata?.reportingPeriod || 'לא צוין',
            sourceType: 'mixed',
            processingTimestamp: new Date().toISOString(),
        }
    };
};