<#
.SYNOPSIS
    向本地的文档问答服务提一个问题，并正确显示中文结果。

.DESCRIPTION
    为什么需要这个脚本，而不是直接用 Invoke-RestMethod？

    Windows PowerShell 5.1 收到响应时，如果响应头里没写 charset，
    它会按西欧编码（ISO-8859-1）去解读内容，中文答案就会变成一串乱码。
    服务端返回的内容本身是正确的 UTF-8，问题只出在 PowerShell 这一侧。

    这个脚本拿到原始字节后，自己按 UTF-8 解码，就绕开了这个坑。

.EXAMPLE
    .\scripts\ask.ps1 "退货的运费谁出？"

.EXAMPLE
    .\scripts\ask.ps1 "发票多久能开" -BaseUrl http://127.0.0.1:8000
#>
param(
    # 要问的问题，必填
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Question,

    # 服务地址。本地开发和 docker compose 默认都是这个
    [string] $BaseUrl = "http://127.0.0.1:8000"
)

# 步骤 1：把问题包成 JSON，再转成 UTF-8 字节。
#         直接传字符串的话，PowerShell 5.1 发送时也可能用错编码，
#         服务端收到的中文就是坏的，会报「请求体解析失败」。
$payload = @{ question = $Question } | ConvertTo-Json -Compress
$bytes = [Text.Encoding]::UTF8.GetBytes($payload)

try {
    # 步骤 2：发请求。-UseBasicParsing 表示不用 IE 引擎解析 HTML，更快也更稳。
    $resp = Invoke-WebRequest -Uri "$BaseUrl/v1/ask" -Method Post `
        -ContentType "application/json; charset=utf-8" `
        -Body $bytes -UseBasicParsing

    # 步骤 3：拿原始字节，自己按 UTF-8 解码。这一步就是修乱码的关键。
    $text = [Text.Encoding]::UTF8.GetString($resp.RawContentStream.ToArray())
}
catch {
    # 步骤 4：出错时分两种情况处理。
    if ($null -eq $_.Exception.Response) {
        # 连响应都没有，说明服务根本没起来，或者地址写错了
        Write-Host "连不上服务：$BaseUrl" -ForegroundColor Red
        Write-Host "先确认服务已经启动：uv run fastapi dev app/main.py"
        exit 1
    }
    # 有响应但状态码不是 2xx，把统一格式的错误体打出来
    $status = [int]$_.Exception.Response.StatusCode
    Write-Host "HTTP $status" -ForegroundColor Yellow
    Write-Host $_.ErrorDetails.Message
    exit 1
}

# 步骤 5：解析 JSON，按人能读的格式打印。
$data = $text | ConvertFrom-Json

Write-Host ""
Write-Host "问：$Question" -ForegroundColor Cyan
Write-Host "答：$($data.answer)"
Write-Host ""

if ($data.citations.Count -eq 0) {
    Write-Host "出处：无" -ForegroundColor DarkGray
}
else {
    Write-Host "出处：" -ForegroundColor DarkGray
    foreach ($c in $data.citations) {
        Write-Host ("  [{0}] {1} · {2}  距离={3:N4}" -f $c.no, $c.source, $c.section, $c.score) -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host ("检索 {0} 条，引用 {1} 条，耗时 {2} ms，request_id={3}" -f `
    $data.retrieved, $data.citations.Count, $data.elapsed_ms, $data.request_id) -ForegroundColor DarkGray
