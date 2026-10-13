// Supplementary images help inspect a run but are not inputs to assertions.
export async function saveDiagnosticScreenshot(page, options, report) {
  let crashed = false;
  const onCrash = () => { crashed = true; };
  page.on('crash', onCrash);
  try {
    await page.screenshot({ ...options, timeout: 15000 });
    return true;
  } catch (error) {
    const browser = page.context().browser();
    if (error.name !== 'TimeoutError' || page.isClosed() || crashed ||
        (browser && !browser.isConnected())) throw error;
    const warning = `Diagnostic screenshot timed out: ${options.path}`;
    (report.artifact_warnings ??= []).push(warning);
    console.warn(warning);
    return false;
  } finally {
    page.off('crash', onCrash);
  }
}
