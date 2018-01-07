<?xml version="1.0"?>

<xsl:stylesheet version="3.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >
<xsl:output method="text" encoding="utf-8"/>

<xsl:strip-space elements="*" />

<xsl:param name="pos_tags"/>
<xsl:variable name="target" select="@lemma"/>

<xsl:template match="w">
  <xsl:variable name="lemma" select="@lemma"/>
  <!--xsl:variable name="token" select="translate($lemma,'|','')"/-->
  <xsl:variable name="token" select="substring-before(substring-after($lemma,'|'),'|')"/>
  <xsl:variable name="content" select="text()"/>
  <xsl:if test="contains($pos_tags,concat('|', @pos, '|'))">
    <xsl:choose>
        <xsl:when test="$token != ''"><xsl:value-of select="$token"/></xsl:when>
        <xsl:otherwise>{<xsl:value-of select="$content"/>}</xsl:otherwise>
    </xsl:choose>
    <xsl:text>&#160;</xsl:text>
  </xsl:if>
</xsl:template>

</xsl:stylesheet>