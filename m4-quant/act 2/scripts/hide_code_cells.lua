-- Lua filter for Pandoc: Suppress input code from notebook code cells
-- while preserving all cell outputs (tables, charts, stdout)
-- and retaining Markdown code blocks in appendices.

function Div(div)
  if div.classes:includes("cell") and div.classes:includes("code") then
    local new_content = {}
    for _, item in ipairs(div.content) do
      if item.t == "Div" and item.classes:includes("output") then
        table.insert(new_content, item)
      end
    end
    div.content = new_content
    return div
  end
end

function Image(img)
  img.attributes['width'] = '75%'
  return {
    pandoc.RawInline('latex', '{\\centering\n'),
    img,
    pandoc.RawInline('latex', '\n\\par}')
  }
end
